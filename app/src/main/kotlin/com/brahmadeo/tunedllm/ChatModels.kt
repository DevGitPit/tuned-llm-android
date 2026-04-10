package com.brahmadeo.tunedllm

enum class Role {
    USER, ASSISTANT
}

data class ChatMessage(
    val id: String,
    val role: Role,
    val content: String
)

data class ChatState(
    val messages: List<ChatMessage> = emptyList(),
    val isGenerating: Boolean = false,
    val isModelLoaded: Boolean = false,
    val isCopying: Boolean = false,
    val copyProgress: Float = 0f,
    val chatTemplate: String = "<start_of_turn>user\n{{prompt}}<end_of_turn>\n<start_of_turn>model\n",
    val error: String? = null
)
