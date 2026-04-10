package com.brahmadeo.tunedllm

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    private val viewModel: LlmViewModel by viewModels()

    private val modelPicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { viewModel.loadSelectedModel(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        viewModel.startAndBindService()

        setContent {
            TunedLLMTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val state by viewModel.uiState.collectAsState()
                    
                    if (state.isModelLoaded) {
                        ChatScreen(viewModel = viewModel)
                    } else {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.Center
                            ) {
                                if (state.isCopying) {
                                    Text("Copying model to internal storage...", style = MaterialTheme.typography.bodyLarge)
                                    Spacer(modifier = Modifier.height(16.dp))
                                    LinearProgressIndicator(
                                        progress = state.copyProgress,
                                        modifier = Modifier.width(200.dp)
                                    )
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text("${(state.copyProgress * 100).toInt()}%")
                                } else {
                                    Button(onClick = { modelPicker.launch("*/*") }) {
                                        Text("Select GGUF Model")
                                    }
                                }
                                
                                if (state.error != null) {
                                    Spacer(modifier = Modifier.height(16.dp))
                                    Text(
                                        text = state.error!!,
                                        color = MaterialTheme.colorScheme.error,
                                        modifier = Modifier.padding(horizontal = 32.dp)
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
