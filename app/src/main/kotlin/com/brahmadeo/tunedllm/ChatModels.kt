package com.brahmadeo.tunedllm

enum class Role {
    USER, ASSISTANT
}

data class ChatMessage(
    val id: String,
    val role: Role,
    val content: String
)

data class ChatSession(
    val id: String,
    val title: String,
    val messages: List<ChatMessage> = emptyList()
)

data class ChatState(
    val sessions: List<ChatSession> = emptyList(),
    val currentSessionId: String? = null,
    val isGenerating: Boolean = false,
    val currentTps: Float? = null,
    val isModelLoaded: Boolean = false,
    val isAutoLoading: Boolean = false,
    val lastModelPath: String? = null,
    val modelName: String? = null,
    val isCopying: Boolean = false,
    val copyProgress: Float = 0f,
    val chatTemplate: String = "<start_of_turn>user\n{{prompt}}<end_of_turn>\n<start_of_turn>model\n",
    val error: String? = null
) {
    val messages: List<ChatMessage>
        get() = sessions.find { it.id == currentSessionId }?.messages ?: emptyList()
}
