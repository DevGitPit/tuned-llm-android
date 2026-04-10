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

    private var llmService: LlmService? = null
    private var isBound = false

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
        
        val userMsgId = UUID.randomUUID().toString()
        val assistantMsgId = UUID.randomUUID().toString()

        val userMsg = ChatMessage(userMsgId, Role.USER, prompt)
        
        // Clean current messages of any leaked tokens just in case
        val sanitizedMessages = _uiState.value.messages.map { 
            it.copy(content = it.content
                .replace("<end_of_turn>", "")
                .replace("<start_of_turn>", "")
                .replace("<eos>", "")
                .trim())
        }

        val history = sanitizedMessages + userMsg
        val assistantMsg = ChatMessage(assistantMsgId, Role.ASSISTANT, "")

        _uiState.update { it.copy(
            messages = history + assistantMsg,
            isGenerating = true,
            error = null
        ) }

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
                        val lastMessage = state.messages.lastOrNull()
                        if (lastMessage?.id == assistantMsgId) {
                            val updatedMessages = state.messages.dropLast(1) + lastMessage.copy(content = lastMessage.content + token)
                            state.copy(messages = updatedMessages)
                        } else {
                            state
                        }
                    }
                }

                override fun onComplete() {
                    _uiState.update { it.copy(isGenerating = false) }
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
