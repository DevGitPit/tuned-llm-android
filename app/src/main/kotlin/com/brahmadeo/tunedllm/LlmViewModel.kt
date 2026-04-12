package com.brahmadeo.tunedllm

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.net.Uri
import android.os.IBinder
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID

class LlmViewModel(application: Application) : AndroidViewModel(application) {
    private val _uiState = MutableStateFlow(ChatState())
    val uiState: StateFlow<ChatState> = _uiState.asStateFlow()

    private val db = ChatDatabase.getDatabase(application)
    private val chatDao = db.chatDao()

    private var llmService: LlmService? = null
    private var isBound = false

    init {
        viewModelScope.launch {
            chatDao.getAllSessionsWithMessages().collect { sessionsWithMessages ->
                val sessions = sessionsWithMessages.map { swm ->
                    ChatSession(
                        id = swm.session.id,
                        title = swm.session.title,
                        messages = swm.messages
                            .sortedBy { it.timestamp }
                            .map { msg ->
                                ChatMessage(
                                    id = msg.id,
                                    role = Role.valueOf(msg.role),
                                    content = msg.content
                                )
                            }
                    )
                }
                
                _uiState.update { state ->
                    val mergedSessions = if (state.isGenerating && state.currentSessionId != null) {
                        sessions.map { session ->
                            if (session.id == state.currentSessionId) {
                                val currentAssistantMsg = state.messages.lastOrNull { it.role == Role.ASSISTANT }
                                if (currentAssistantMsg != null && session.messages.none { it.id == currentAssistantMsg.id }) {
                                    session.copy(messages = session.messages + currentAssistantMsg)
                                } else {
                                    session
                                }
                            } else {
                                session
                            }
                        }
                    } else {
                        sessions
                    }
                    state.copy(sessions = mergedSessions)
                }
                
                // If no current session selected, pick the first one from DB or create new
                if (_uiState.value.currentSessionId == null) {
                    if (sessions.isNotEmpty()) {
                        _uiState.update { it.copy(currentSessionId = sessions.first().id) }
                    } else {
                        createNewSession()
                    }
                }
            }
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as LlmService.LlmBinder
            llmService = binder.getService()
            isBound = true
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            isBound = false
            llmService = null
        }
    }

    fun startAndBindService() {
        val intent = Intent(getApplication(), LlmService::class.java)
        ContextCompat.startForegroundService(getApplication(), intent)
        getApplication<Application>().bindService(intent, serviceConnection, Context.BIND_IMPORTANT)
    }

    private fun ensureSession() {
        if (_uiState.value.sessions.isEmpty()) {
            createNewSession()
        }
    }

    fun createNewSession() {
        val newId = UUID.randomUUID().toString()
        _uiState.update { it.copy(currentSessionId = newId) }
        viewModelScope.launch {
            chatDao.insertSession(SessionEntity(newId, "New Chat"))
        }
    }

    fun selectSession(sessionId: String) {
        if (_uiState.value.currentSessionId == sessionId) return
        _uiState.update { it.copy(currentSessionId = sessionId) }
    }

    fun deleteSession(sessionId: String) {
        val currentId = _uiState.value.currentSessionId
        _uiState.update { state ->
            val updatedSessions = state.sessions.filter { it.id != sessionId }
            var nextSessionId = state.currentSessionId
            if (state.currentSessionId == sessionId) {
                nextSessionId = updatedSessions.lastOrNull()?.id
            }
            state.copy(currentSessionId = nextSessionId)
        }
        viewModelScope.launch {
            chatDao.deleteSession(sessionId)
        }
    }

    fun updateChatTemplate(template: String) {
        _uiState.update { it.copy(chatTemplate = template) }
    }

    fun loadSelectedModel(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { it.copy(isCopying = true, copyProgress = 0f, error = null) }
            
            val manager = llmService?.llmManager
            if (manager == null) {
                _uiState.update { it.copy(isCopying = false, error = "Service not bound") }
                return@launch
            }

            val path = manager.copyModelToInternalStorage(uri) { progress ->
                _uiState.update { it.copy(copyProgress = progress) }
            }

            if (path != null) {
                _uiState.update { it.copy(isCopying = false) }
                viewModelScope.launch(Dispatchers.IO) {
                    val result = manager.loadModelFromPath(path)
                    if (result.isSuccess) {
                        _uiState.update { it.copy(isModelLoaded = true, error = null) }
                    } else {
                        _uiState.update { it.copy(isModelLoaded = false, error = result.exceptionOrNull()?.message ?: "Failed to load model") }
                    }
                }
            } else {
                _uiState.update { it.copy(isCopying = false, error = "Failed to copy model to internal storage") }
            }
        }
    }

    fun generate(prompt: String) {
        if (!isBound || !_uiState.value.isModelLoaded) return
        
        ensureSession()
        val currentSessionId = _uiState.value.currentSessionId ?: return

        val userMsgId = UUID.randomUUID().toString()
        val assistantMsgId = UUID.randomUUID().toString()

        val userMsg = ChatMessage(userMsgId, Role.USER, prompt)
        
        // Clean current messages of any leaked tokens just in case
        val sanitizedMessages = _uiState.value.messages.map { 
            it.copy(content = it.content
                .replace("<end_of_turn>", "")
                .replace("<start_of_turn>", "")
                .replace("</start_of_turn>", "")
                .replace("<eos>", "")
                .trim())
        }

        val history = sanitizedMessages + userMsg
        val assistantMsg = ChatMessage(assistantMsgId, Role.ASSISTANT, "")

        _uiState.update { state ->
            val updatedSessions = state.sessions.map { session ->
                if (session.id == currentSessionId) {
                    val newTitle = if ((session.title == "New Chat" || session.title.isBlank()) && prompt.isNotBlank()) {
                        prompt.take(30).trim().plus(if (prompt.length > 30) "..." else "")
                    } else {
                        session.title
                    }
                    session.copy(title = newTitle, messages = history + assistantMsg)
                } else {
                    session
                }
            }
            state.copy(
                sessions = updatedSessions,
                isGenerating = true,
                error = null
            )
        }

        viewModelScope.launch {
            chatDao.insertMessage(MessageEntity(userMsgId, currentSessionId, Role.USER.name, prompt))
            // Re-read current title after update above
            val currentTitle = _uiState.value.sessions.find { it.id == currentSessionId }?.title
            if (currentTitle != null) {
                chatDao.updateSession(SessionEntity(currentSessionId, currentTitle))
            }
        }

        // Build the Gemma chat template with full history
        val promptBuilder = StringBuilder()
        for (msg in history) {
            val roleTag = if (msg.role == Role.USER) "user" else "model"
            promptBuilder.append("<start_of_turn>$roleTag\n${msg.content.trim()}<end_of_turn>\n")
        }
        promptBuilder.append("<start_of_turn>model\n")
        
        val formattedPrompt = promptBuilder.toString()
        Log.d("LlmViewModel", "Sending prompt to JNI: $formattedPrompt")

        viewModelScope.launch(Dispatchers.IO) {
            llmService?.llmManager?.generate(formattedPrompt, object : LlmCallback {
                override fun onToken(token: String) {
                    _uiState.update { state ->
                        val updatedSessions = state.sessions.map { session ->
                            if (session.id == currentSessionId) {
                                val lastMsg = session.messages.lastOrNull()
                                if (lastMsg?.id == assistantMsgId) {
                                    val updatedMessages = session.messages.dropLast(1) + lastMsg.copy(content = lastMsg.content + token)
                                    session.copy(messages = updatedMessages)
                                } else {
                                    session
                                }
                            } else {
                                session
                            }
                        }
                        state.copy(sessions = updatedSessions)
                    }
                }

                override fun onComplete() {
                    val finalMsg = _uiState.value.messages.lastOrNull { it.id == assistantMsgId }
                    if (finalMsg != null) {
                        viewModelScope.launch {
                            chatDao.insertMessage(MessageEntity(assistantMsgId, currentSessionId, Role.ASSISTANT.name, finalMsg.content))
                            _uiState.update { it.copy(isGenerating = false) }
                        }
                    } else {
                        _uiState.update { it.copy(isGenerating = false) }
                    }
                }

                override fun onError(message: String) {
                    _uiState.update { it.copy(isGenerating = false, error = message) }
                }
            })
        }
    }

    fun stopGeneration() {
        llmService?.llmManager?.stop()
        _uiState.update { it.copy(isGenerating = false) }
    }

    override fun onCleared() {
        super.onCleared()
        if (isBound) {
            getApplication<Application>().unbindService(serviceConnection)
            isBound = false
        }
    }
}
