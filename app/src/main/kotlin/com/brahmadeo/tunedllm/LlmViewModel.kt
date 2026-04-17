package com.brahmadeo.tunedllm

import android.app.Application
import android.content.*
import android.net.Uri
import android.os.IBinder
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Toast
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.util.UUID

class LlmViewModel(application: Application) : AndroidViewModel(application) {
    private val prefs = application.getSharedPreferences("tuned_llm_prefs", Context.MODE_PRIVATE)
    private val database = ChatDatabase.getDatabase(application)
    private val chatDao = database.chatDao()

    private val _uiState = MutableStateFlow(ChatState())
    val uiState: StateFlow<ChatState> = _uiState.asStateFlow()

    private val _ramInfo = MutableStateFlow<RamInfo?>(null)
    val ramInfo: StateFlow<RamInfo?> = _ramInfo.asStateFlow()

    private var selectedUri: Uri? = null
    private val _selectedFileSize = MutableStateFlow<Long?>(null)
    val selectedFileSize: StateFlow<Long?> = _selectedFileSize.asStateFlow()

    val isModelEverLoaded: Boolean
        get() = prefs.getBoolean("model_ever_loaded", false)

    private var llmService: LlmService? = null
    private var isBound = false
    private var lastActiveSessionId: String? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as LlmService.LlmBinder
            llmService = binder.getService()
            isBound = true
            
            // Auto-load if we have a saved path
            val path = _uiState.value.lastModelPath
            val name = _uiState.value.modelName
            if (path != null && name != null && _uiState.value.isAutoLoading) {
                loadModelFromPath(path, name)
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            llmService = null
            isBound = false
        }
    }

    init {
        refreshRamInfo(application)
        val savedModelPath = prefs.getString("last_model_path", null)
        val savedModelName = prefs.getString("last_model_name", null)
        _uiState.update { it.copy(
            lastModelPath = savedModelPath,
            modelName = savedModelName,
            chatTemplate = ChatTemplate.fromModelName(savedModelName),
            isAutoLoading = savedModelPath != null
        ) }

        viewModelScope.launch {
            chatDao.getAllSessionsWithMessages().collect { sessionsWithMessages ->
                val sessions = sessionsWithMessages.map { it.toChatSession() }
                _uiState.update { state ->
                    val currentId = state.currentSessionId ?: sessions.lastOrNull()?.id
                    state.copy(sessions = sessions, currentSessionId = currentId)
                }
            }
        }

        val intent = Intent(application, LlmService::class.java)
        application.startService(intent)
        application.bindService(intent, serviceConnection, Context.BIND_IMPORTANT)
    }

    private fun ensureSession() {
        if (_uiState.value.sessions.isEmpty()) {
            createNewSession()
        }
    }

    fun createNewSession() {
        viewModelScope.launch {
            llmService?.llmManager?.clearContext()
            val newId = UUID.randomUUID().toString()
            _uiState.update { it.copy(currentSessionId = newId) }
            chatDao.insertSession(SessionEntity(newId, "New Chat"))
        }
    }

    fun selectSession(sessionId: String) {
        if (_uiState.value.currentSessionId == sessionId) return
        viewModelScope.launch {
            llmService?.llmManager?.clearContext()
            _uiState.update { it.copy(currentSessionId = sessionId) }
        }
    }

    fun deleteSession(sessionId: String) {
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

    fun updateChatTemplate(template: ChatTemplate) {
        _uiState.update { it.copy(chatTemplate = template) }
    }

    fun updateGenerationMode(mode: GenerationMode) {
        val isGemma4 = _uiState.value.chatTemplate == ChatTemplate.GEMMA4
        val newConfig = when (mode) {
            GenerationMode.GENERAL -> {
                if (isGemma4) GenerationConfig(mode = mode, temperature = 1.0f, topP = 0.95f, topK = 64, presencePenalty = 1.0f)
                else GenerationConfig(mode = mode, temperature = 0.7f, topP = 0.8f, topK = 20, presencePenalty = 1.5f)
            }
            GenerationMode.CODING -> {
                if (isGemma4) GenerationConfig(mode = mode, temperature = 1.0f, topP = 0.95f, topK = 64, presencePenalty = 1.0f)
                else GenerationConfig(mode = mode, temperature = 0.6f, topP = 0.95f, topK = 20, presencePenalty = 0.0f)
            }
            GenerationMode.REASONING -> {
                if (isGemma4) GenerationConfig(mode = mode, temperature = 1.0f, topP = 0.95f, topK = 64, presencePenalty = 1.0f)
                else GenerationConfig(mode = mode, temperature = 1.0f, topP = 0.95f, topK = 40, presencePenalty = 1.5f)
            }
        }
        _uiState.update { it.copy(config = newConfig) }
    }

    fun toggleThinking(enabled: Boolean) {
        _uiState.update { it.copy(config = it.config.copy(enableThinking = enabled)) }
    }

    fun loadSelectedModel(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { it.copy(isCopying = true, copyProgress = 0f, error = null) }
            
            val manager = llmService?.llmManager
            if (manager == null) {
                _uiState.update { it.copy(isCopying = false, error = "Service not bound") }
                return@launch
            }

            var originalName = "Model"
            getApplication<Application>().contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (cursor.moveToFirst()) {
                    originalName = cursor.getString(nameIndex)
                }
            }

            val path = manager.copyModelToInternalStorage(uri) { progress ->
                _uiState.update { it.copy(copyProgress = progress) }
            }

            if (path != null) {
                _uiState.update { it.copy(isCopying = false) }
                loadModelFromPath(path, originalName)
            } else {
                _uiState.update { it.copy(isCopying = false, error = "Failed to copy model to internal storage") }
            }
        }
    }

    private fun loadModelFromPath(path: String, modelName: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val manager = llmService?.llmManager ?: return@launch
            val result = manager.loadModelFromPath(path)
            if (result.isSuccess) {
                prefs.edit()
                    .putString("last_model_path", path)
                    .putString("last_model_name", modelName)
                    .putBoolean("model_ever_loaded", true)
                    .apply()
                _uiState.update { it.copy(
                    isModelLoaded = true, 
                    lastModelPath = path, 
                    modelName = modelName,
                    chatTemplate = ChatTemplate.fromModelName(modelName),
                    isAutoLoading = false,
                    error = null
                ) }
            } else {
                _uiState.update { it.copy(
                    isModelLoaded = false, 
                    isAutoLoading = false,
                    error = result.exceptionOrNull()?.message ?: "Failed to load model"
                ) }
            }
        }
    }

    fun unloadModel() {
        llmService?.llmManager?.unloadModel()
        _uiState.update { it.copy(isModelLoaded = false, lastModelPath = null, modelName = null) }
        prefs.edit().remove("last_model_path").remove("last_model_name").apply()
    }

    fun refreshRamInfo(context: Context) {
        _ramInfo.value = getDeviceRamInfo(context)
    }

    fun onFileSelected(uri: Uri, context: Context) {
        selectedUri = uri
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
            if (cursor.moveToFirst()) {
                _selectedFileSize.value = cursor.getLong(sizeIndex)
            }
        }
    }

    fun confirmAndLoad() {
        val uri = selectedUri ?: return
        loadSelectedModel(uri)
    }

    fun copyToClipboard(text: String) {
        val clipboard = getApplication<Application>().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("Tuned LLM", text)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(getApplication(), "Copied to clipboard", Toast.LENGTH_SHORT).show()
    }

    fun generate(prompt: String) {
        if (!isBound || !_uiState.value.isModelLoaded) return
        
        ensureSession()
        val currentSessionId = _uiState.value.currentSessionId ?: return

        // If we switched sessions, force a clear
        if (lastActiveSessionId != null && lastActiveSessionId != currentSessionId) {
            llmService?.llmManager?.clearContext()
        }
        lastActiveSessionId = currentSessionId

        val userMsgId = UUID.randomUUID().toString()
        val assistantMsgId = UUID.randomUUID().toString()

        val userMsg = ChatMessage(userMsgId, Role.USER, prompt)
        
        val template = _uiState.value.chatTemplate
        val config = _uiState.value.config

        // Clean and potentially PRUNE thinking tags from history
        val sanitizedMessages = _uiState.value.messages.map { msg ->
            var content = msg.content
                .replace("<end_of_turn>", "")
                .replace("<start_of_turn>", "")
                .replace("</start_of_turn>", "")
                .replace("<eos>", "")
                .replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
                .replace("<|endoftext|>", "")
                .replace("<|eot_id|>", "")
                .replace("<|start_header_id|>", "")
                .replace("<|end_header_id|>", "")
                .trim()
            
            // SMART PRUNING: Only for history
            if (msg.role == Role.ASSISTANT && template.shouldPruneThinkingFromHistory) {
                if (template.thinkStartTag != null && template.thinkEndTag != null) {
                    val sIdx = content.indexOf(template.thinkStartTag)
                    val eIdx = content.indexOf(template.thinkEndTag)
                    if (sIdx != -1 && eIdx != -1 && eIdx > sIdx) {
                        val before = content.substring(0, sIdx)
                        val after = content.substring(eIdx + template.thinkEndTag.length)
                        content = (before + after).trim()
                    }
                }
            }
            msg.copy(content = content)
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
            val currentTitle = _uiState.value.sessions.find { it.id == currentSessionId }?.title
            if (currentTitle != null) {
                chatDao.updateSession(SessionEntity(currentSessionId, currentTitle))
            }
        }

        val promptBuilder = StringBuilder()
        
        // Handle GEMMA4 special thinking trigger in system prompt
        if (config.enableThinking && template == ChatTemplate.GEMMA4) {
            promptBuilder.append("<|turn>system\n<|think|>You are a helpful assistant with reasoning capabilities.<turn|>\n")
        }

        for (msg in history) {
            val roleTag = if (msg.role == Role.USER) template.userRole else template.assistantRole
            promptBuilder.append("${template.prefix}$roleTag${template.roleSuffix}${msg.content}${template.eot}")
        }
        promptBuilder.append("${template.prefix}${template.assistantRole}${template.roleSuffix}")
        
        // Pre-inject thinking tag if enabled to force the model to start reasoning
        if (config.enableThinking && template.thinkStartTag != null) {
            if (_uiState.value.modelName?.contains("qwen", true) == true || 
                _uiState.value.modelName?.contains("gemma-4", true) == true ||
                _uiState.value.modelName?.contains("gemma4", true) == true) {
                promptBuilder.append(template.thinkStartTag)
                // Gemma 4 specific suffix for thought channel
                if (template == ChatTemplate.GEMMA4) {
                    promptBuilder.append("\n")
                }
            }
        }
        
        val formattedPrompt = promptBuilder.toString()
        Log.d("LlmViewModel", "Sending prompt to JNI: $formattedPrompt")

        var tokenCount = 0
        var startTime = 0L

        viewModelScope.launch(Dispatchers.IO) {
            llmService?.llmManager?.generate(
                formattedPrompt, 
                template.stopStrings.toTypedArray(),
                config.temperature,
                config.topP,
                config.topK,
                config.presencePenalty,
                object : LlmCallback {
                    override fun onToken(token: String) {
                        if (startTime == 0L) startTime = System.currentTimeMillis()
                        tokenCount++
                        val elapsed = (System.currentTimeMillis() - startTime) / 1000f
                        val tps = if (elapsed > 0) tokenCount / elapsed else 0f

                        _uiState.update { state ->
                            val updatedSessions = state.sessions.map { session ->
                                if (session.id == currentSessionId) {
                                    val lastMsg = session.messages.lastOrNull()
                                    if (lastMsg?.id == assistantMsgId) {
                                        session.copy(messages = session.messages.dropLast(1) + lastMsg.copy(content = lastMsg.content + token))
                                    } else session
                                } else session
                            }
                            state.copy(sessions = updatedSessions, currentTps = tps, isGenerating = true)
                        }
                    }

                    override fun onComplete() {
                        val finalMsg = _uiState.value.sessions.find { it.id == currentSessionId }?.messages?.lastOrNull { it.id == assistantMsgId }
                        if (finalMsg != null) {
                            viewModelScope.launch {
                                chatDao.insertMessage(MessageEntity(assistantMsgId, currentSessionId, Role.ASSISTANT.name, finalMsg.content))
                                _uiState.update { it.copy(isGenerating = false, currentTps = null) }
                            }
                        } else {
                            _uiState.update { it.copy(isGenerating = false, currentTps = null) }
                        }
                    }

                    override fun onError(message: String) {
                        Log.e("LlmViewModel", "Generation error: $message")
                        val isStopped = message.contains("stopped", true)
                        _uiState.update { state -> state.copy(isGenerating = false, error = if (isStopped) null else message, currentTps = null) }
                    }
                }
            )
        }
    }

    fun stopGeneration() {
        llmService?.llmManager?.stop()
        _uiState.update { it.copy(isGenerating = false, currentTps = null) }
    }

    override fun onCleared() {
        super.onCleared()
        if (isBound) {
            getApplication<Application>().unbindService(serviceConnection)
            isBound = false
        }
    }
}
