package com.brahmadeo.tunedllm

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import dev.jeziellago.compose.markdowntext.MarkdownText

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: LlmViewModel) {
    val state by viewModel.uiState.collectAsState()
    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }
    var showTemplateDialog by remember { mutableStateOf(false) }

    val isAtBottom by remember {
        derivedStateOf {
            val layoutInfo = listState.layoutInfo
            val visibleItemsInfo = layoutInfo.visibleItemsInfo
            if (layoutInfo.totalItemsCount == 0) {
                true
            } else {
                val lastVisibleItem = visibleItemsInfo.lastOrNull()
                lastVisibleItem?.index == layoutInfo.totalItemsCount - 1
            }
        }
    }

    // Improved auto-scrolling: only scroll if at bottom or when a new message starts
    LaunchedEffect(state.messages.size, state.messages.lastOrNull()?.content) {
        if (isAtBottom || state.isGenerating) {
            if (state.messages.isNotEmpty()) {
                listState.animateScrollToItem(state.messages.size - 1)
            }
        }
    }

    if (showTemplateDialog) {
        var templateText by remember { mutableStateOf(state.chatTemplate) }
        AlertDialog(
            onDismissRequest = { showTemplateDialog = false },
            title = { Text("Edit Chat Template") },
            text = {
                Column {
                    Text("Use {{prompt}} as placeholder for user message.", style = MaterialTheme.typography.bodySmall)
                    Spacer(modifier = Modifier.height(8.dp))
                    TextField(
                        value = templateText,
                        onValueChange = { templateText = it },
                        modifier = Modifier.fillMaxWidth(),
                        maxLines = 10
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.updateChatTemplate(templateText)
                    showTemplateDialog = false
                }) {
                    Text("Save")
                }
            },
            dismissButton = {
                TextButton(onClick = { showTemplateDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Tuned LLM Chat") },
                actions = {
                    IconButton(onClick = { showTemplateDialog = true }) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.primary
                )
            )
        },
        bottomBar = {
            Surface(tonalElevation = 8.dp) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                ) {
                    if (state.error != null) {
                        Text(
                            text = "Error: ${state.error}",
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        TextField(
                            value = inputText,
                            onValueChange = { inputText = it },
                            modifier = Modifier.weight(1f),
                            placeholder = { Text("Enter prompt...") },
                            maxLines = 4
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Button(
                            onClick = {
                                if (state.isGenerating) {
                                    viewModel.stopGeneration()
                                } else {
                                    if (inputText.isNotBlank()) {
                                        viewModel.generate(inputText)
                                        inputText = ""
                                    }
                                }
                            },
                            enabled = state.isModelLoaded,
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text(if (state.isGenerating) "Stop" else "Send")
                        }
                    }
                }
            }
        }
    ) { paddingValues ->
        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 8.dp),
            contentPadding = PaddingValues(vertical = 8.dp)
        ) {
            items(state.messages, key = { it.id }) { message ->
                ChatMessageItem(message)
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

@Composable
fun ChatMessageItem(message: ChatMessage) {
    val isUser = message.role == Role.USER
    val alignment = if (isUser) Alignment.End else Alignment.Start
    val containerColor = if (isUser) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.secondaryContainer
    }
    val contentColor = if (isUser) {
        MaterialTheme.colorScheme.onPrimary
    } else {
        MaterialTheme.colorScheme.onSecondaryContainer
    }

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        Surface(
            color = containerColor,
            contentColor = contentColor,
            shape = RoundedCornerShape(
                topStart = 12.dp,
                topEnd = 12.dp,
                bottomStart = if (isUser) 12.dp else 0.dp,
                bottomEnd = if (isUser) 0.dp else 12.dp
            ),
            tonalElevation = 2.dp
        ) {
            MarkdownText(
                markdown = message.content,
                modifier = Modifier.padding(12.dp),
                style = MaterialTheme.typography.bodyMedium.copy(color = contentColor),
                syntaxHighlightColor = if (isSystemInDarkTheme()) Color(0xFF2D2D2D) else Color(0xFFF5F5F5)
            )
        }
        Text(
            text = if (isUser) "You" else "Assistant",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.outline,
            modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp)
        )
    }
}
