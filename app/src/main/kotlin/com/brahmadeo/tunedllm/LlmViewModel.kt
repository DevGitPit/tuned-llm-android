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
        val savedNCtx = prefs.getInt("n_ctx", 8192)
        _uiState.update { it.copy(
            lastModelPath = savedModelPath,
            modelName = savedModelName,
            chatTemplate = ChatTemplate.fromModelName(savedModelName),
            isAutoLoading = savedModelPath != null,
            config = GenerationConfig(nCtx = savedNCtx)
        ) }

        viewModelScope.launch {
            chatDao.getAllSessionsWithMessages().collect { sessionsWithMessages ->
                val dbSessions = sessionsWithMessages.map { it.toChatSession() }
                _uiState.update { state ->
                    val currentId = state.currentSessionId ?: dbSessions.lastOrNull()?.id
                    
                    val mergedSessions = if (state.isGenerating && state.currentSessionId != null) {
                        dbSessions.map { dbSession ->
                            if (dbSession.id == state.currentSessionId) {
                                val memorySession = state.sessions.find { it.id == dbSession.id }
                                if (memorySession != null && memorySession.messages.size > dbSession.messages.size) {
                                    memorySession
                                } else dbSession
                            } else dbSession
                        }
                    } else dbSessions

                    state.copy(sessions = mergedSessions, currentSessionId = currentId)
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
        stopGeneration()
        viewModelScope.launch {
            llmService?.llmManager?.clearContext()
            val newId = UUID.randomUUID().toString()
            _uiState.update { it.copy(currentSessionId = newId, lastPpStatus = null, lastTgStatus = null, lastGenerationStatus = null, contextProgress = 0f) }
            chatDao.insertSession(SessionEntity(newId, "New Chat"))
        }
    }

    fun selectSession(sessionId: String) {
        if (_uiState.value.currentSessionId == sessionId) return
        stopGeneration()
        viewModelScope.launch {
            llmService?.llmManager?.clearContext()
            _uiState.update { it.copy(currentSessionId = sessionId, lastPpStatus = null, lastTgStatus = null, lastGenerationStatus = null, contextProgress = 0f) }
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

    fun updateNCtx(nCtx: Int) {
        prefs.edit().putInt("n_ctx", nCtx).apply()
        _uiState.update { it.copy(config = it.config.copy(nCtx = nCtx)) }
        val path = _uiState.value.lastModelPath
        val name = _uiState.value.modelName
        if (path != null && name != null && _uiState.value.isModelLoaded) {
            stopGeneration()
            loadModelFromPath(path, name)
        }
    }

    fun updateGenerationMode(mode: GenerationMode) {
        val isGemma4 = _uiState.value.chatTemplate == ChatTemplate.GEMMA4
        val currentNCtx = _uiState.value.config.nCtx
        val newConfig = when (mode) {
            GenerationMode.GENERAL -> {
                if (isGemma4) GenerationConfig(mode = mode, temperature = 1.0f, topP = 0.95f, topK = 64, minP = 0.05f, presencePenalty = 1.0f, nCtx = currentNCtx)
                else GenerationConfig(mode = mode, temperature = 0.7f, topP = 0.8f, topK = 20, minP = 0.05f, presencePenalty = 1.5f, nCtx = currentNCtx)
            }
            GenerationMode.CODING -> {
                if (isGemma4) GenerationConfig(mode = mode, temperature = 1.0f, topP = 0.95f, topK = 64, minP = 0.05f, presencePenalty = 1.0f, nCtx = currentNCtx)
                else GenerationConfig(mode = mode, temperature = 0.6f, topP = 0.95f, topK = 20, minP = 0.05f, presencePenalty = 0.0f, nCtx = currentNCtx)
            }
            GenerationMode.REASONING -> {
                if (isGemma4) GenerationConfig(mode = mode, temperature = 1.0f, topP = 0.95f, topK = 64, minP = 0.05f, presencePenalty = 1.0f, nCtx = currentNCtx)
                else GenerationConfig(mode = mode, temperature = 0.7f, topP = 0.8f, topK = 40, minP = 0.1f, presencePenalty = 1.5f, repetitionPenalty = 1.2f, nCtx = currentNCtx)
            }
        }
        _uiState.update { it.copy(config = newConfig) }
    }

    fun toggleThinking(enabled: Boolean) {
        _uiState.update { it.copy(config = it.config.copy(enableThinking = enabled)) }
    }

    fun loadSelectedModel(uri: Uri) {
        stopGeneration()
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
                if (cursor.moveToFirst()) originalName = cursor.getString(nameIndex)
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
            val nCtx = _uiState.value.config.nCtx
            val result = manager.loadModelFromPath(path, nCtx = nCtx)
            if (result.isSuccess) {
                prefs.edit()
                    .putString("last_model_path", path)
                    .putString("last_model_name", modelName)
                    .putBoolean("model_ever_loaded", true)
                    .apply()
                _uiState.update { it.copy(
                    isModelLoaded = true, lastModelPath = path, modelName = modelName,
                    chatTemplate = ChatTemplate.fromModelName(modelName), isAutoLoading = false, error = null
                ) }
            } else {
                _uiState.update { it.copy(isModelLoaded = false, isAutoLoading = false, error = result.exceptionOrNull()?.message ?: "Failed to load model") }
            }
        }
    }

    fun unloadModel() {
        llmService?.llmManager?.unloadModel()
        _uiState.update { it.copy(isModelLoaded = false, lastModelPath = null, modelName = null) }
        prefs.edit().remove("last_model_path").remove("last_model_name").apply()
    }

    fun refreshRamInfo(context: Context) { _ramInfo.value = getDeviceRamInfo(context) }

    fun onFileSelected(uri: Uri, context: Context) {
        stopGeneration()
        selectedUri = uri
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
            if (cursor.moveToFirst()) _selectedFileSize.value = cursor.getLong(sizeIndex)
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
        if (lastActiveSessionId != null && lastActiveSessionId != currentSessionId) llmService?.llmManager?.clearContext()
        lastActiveSessionId = currentSessionId

        val userMsgId = UUID.randomUUID().toString()
        val assistantMsgId = UUID.randomUUID().toString()
        val userMsg = ChatMessage(userMsgId, Role.USER, prompt)
        val template = _uiState.value.chatTemplate
        val config = _uiState.value.config

        val sanitizedMessages = _uiState.value.messages.map { msg ->
            var content = msg.content
                .replace("<end_of_turn>", "").replace("<start_of_turn>", "")
                .replace("</start_of_turn>", "").replace("<eos>", "")
                .replace("<|im_start|>", "").replace("<|im_end|>", "")
                .replace("<|endoftext|>", "").replace("<|eot_id|>", "")
                .replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
                .trim()
            if (msg.role == Role.ASSISTANT && template.shouldPruneThinkingFromHistory) {
                if (template.thinkStartTag != null && template.thinkEndTag != null) {
                    val sIdx = content.indexOf(template.thinkStartTag)
                    val eIdx = content.indexOf(template.thinkEndTag)
                    if (sIdx != -1 && eIdx != -1 && eIdx > sIdx) {
                        content = (content.substring(0, sIdx) + content.substring(eIdx + template.thinkEndTag.length)).trim()
                    }
                }
            }
            msg.copy(content = content)
        }

        val history = sanitizedMessages + userMsg
        var initialAssistantContent = ""
        if (config.enableThinking && template.thinkStartTag != null) {
            val mName = _uiState.value.modelName ?: ""
            if (mName.contains("qwen", true) || mName.contains("gemma-4", true) || mName.contains("gemma4", true)) {
                initialAssistantContent = template.thinkStartTag!!
                if (template == ChatTemplate.GEMMA4) initialAssistantContent += "\n"
            }
        }

        val assistantMsg = ChatMessage(assistantMsgId, Role.ASSISTANT, initialAssistantContent)
        _uiState.update { it.copy(
            isGenerating = true, error = null, currentStatus = "Processing...", 
            lastPpStatus = null, lastTgStatus = null, lastGenerationStatus = null,
            sessions = it.sessions.map { s -> if (s.id == currentSessionId) s.copy(messages = history + assistantMsg) else s }
        ) }

        viewModelScope.launch { chatDao.insertMessage(MessageEntity(userMsgId, currentSessionId, Role.USER.name, prompt)) }

        // 1. Generate the dynamic system prompt based on current thinking state
        val currentDateTime = java.text.SimpleDateFormat("EEEE, MMMM dd, yyyy", java.util.Locale.getDefault()).format(java.util.Date())
        val baseContent = "Today is $currentDateTime. Use this as absolute fact. Knowledge cutoff: ${template.knowledgeCutoff}."
        
        val systemContent = if (config.enableThinking) {
            "You are a helpful AI assistant. $baseContent Provide concise reasoning followed by a clear answer."
        } else {
            "You are a helpful AI assistant. $baseContent Provide a direct, concise answer. DO NOT use reasoning tags, internal monologue, or thinking blocks. STRICTLY provide only the final output."
        }

        val systemPrompt = when(template) {
            ChatTemplate.GEMMA4 -> "<bos><|turn>system\n${if (config.enableThinking) "<|think|>" else ""}$systemContent<turn|>\n"
            ChatTemplate.QWEN -> "<bos><|im_start|>system\n$systemContent<|im_end|>\n"
            ChatTemplate.LLAMA3 -> "<bos><|start_header_id|>system<|end_header_id|>\n\n$systemContent<|eot_id|>"
            else -> "<bos>$systemContent\n"
        }

        val promptBuilder = StringBuilder().append(systemPrompt)
        for (msg in history) promptBuilder.append("${template.prefix}${if (msg.role == Role.USER) template.userRole else template.assistantRole}${template.roleSuffix}${msg.content}${template.eot}")
        promptBuilder.append("${template.prefix}${template.assistantRole}${template.roleSuffix}")
        if (config.enableThinking && template.thinkStartTag != null) {
            val mName = _uiState.value.modelName ?: ""
            if (mName.contains("qwen", true) || mName.contains("gemma-4", true) || mName.contains("gemma4", true)) {
                promptBuilder.append(template.thinkStartTag)
                if (template == ChatTemplate.GEMMA4) promptBuilder.append("\n")
            }
        }
        
        val fPrompt = promptBuilder.toString()
        viewModelScope.launch(Dispatchers.IO) {
            llmService?.llmManager?.generate(
                fPrompt, template.stopStrings.toTypedArray(),
                config.temperature, config.topP, config.topK, config.minP, config.presencePenalty, config.repetitionPenalty,
                object : LlmCallback {
                    override fun onStatus(status: String) {
                        _uiState.update { state ->
                            val updatedPp = if (status.startsWith("Processing")) status.substringAfter("Processing... ").trim() else state.lastPpStatus
                            val updatedTg = if (status.startsWith("Generating")) status.substringAfter("Generating... ").trim() else state.lastTgStatus
                            state.copy(currentStatus = status, lastPpStatus = updatedPp, lastTgStatus = updatedTg)
                        }
                    }

                    override fun onToken(token: String) {
                        _uiState.update { state ->
                            val totalTokens = (state.messages.find { it.id == assistantMsgId }?.content?.length ?: 0) / 4 + history.sumOf { it.content.length } / 4
                            val progress = totalTokens.toFloat() / state.config.nCtx
                            val updatedSessions = state.sessions.map { session ->
                                if (session.id == currentSessionId) {
                                    val lastMsg = session.messages.lastOrNull()
                                    if (lastMsg?.id == assistantMsgId) session.copy(messages = session.messages.dropLast(1) + lastMsg.copy(content = lastMsg.content + token))
                                    else session
                                } else session
                            }
                            state.copy(sessions = updatedSessions, isGenerating = true, contextProgress = progress.coerceIn(0f, 1f))
                        }
                    }

                    override fun onComplete() {
                        _uiState.update { state ->
                            val summary = buildString {
                                if (state.lastPpStatus != null) append("PP: ${state.lastPpStatus}")
                                if (state.lastPpStatus != null && state.lastTgStatus != null) append(" | ")
                                if (state.lastTgStatus != null) append("TG: ${state.lastTgStatus}")
                            }
                            state.copy(isGenerating = false, currentStatus = null, lastGenerationStatus = if (summary.isEmpty()) null else summary)
                        }
                        val finalMsg = _uiState.value.sessions.find { it.id == currentSessionId }?.messages?.lastOrNull { it.id == assistantMsgId }
                        if (finalMsg != null) viewModelScope.launch { chatDao.insertMessage(MessageEntity(assistantMsgId, currentSessionId, Role.ASSISTANT.name, finalMsg.content)) }
                    }

                    override fun onError(message: String) {
                        val isStopped = message.contains("stopped", true)
                        _uiState.update { it.copy(isGenerating = false, error = if (isStopped) null else message, currentStatus = null) }
                    }
                }
            )
        }
    }

    fun stopGeneration() {
        llmService?.llmManager?.stop()
        _uiState.update { it.copy(isGenerating = false, currentStatus = null) }
    }

    override fun onCleared() {
        super.onCleared()
        if (isBound) getApplication<Application>().unbindService(serviceConnection)
    }
}
