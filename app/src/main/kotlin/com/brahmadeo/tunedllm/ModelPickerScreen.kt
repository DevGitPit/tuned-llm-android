package com.brahmadeo.tunedllm

import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelPickerScreen(viewModel: LlmViewModel) {
    val state by viewModel.uiState.collectAsState()
    val ramInfo by viewModel.ramInfo.collectAsState()
    val selectedFileSize by viewModel.selectedFileSize.collectAsState()
    val context = LocalContext.current
    var showBottomSheet by remember { mutableStateOf(false) }
    val sheetState = rememberModalBottomSheetState()

    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { viewModel.onFileSelected(it, context) }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Welcome to Tuned LLM",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 24.dp)
            )

            // Section 1: RAM Status Card
            RamStatusCard(ramInfo, onRefresh = { viewModel.refreshRamInfo(context) })

            Spacer(modifier = Modifier.height(24.dp))

            // Section 2: Recommendation Bottom Sheet Trigger
            OutlinedButton(
                onClick = { showBottomSheet = true },
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Default.Info, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Check Model Size Recommendation")
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Section 3: File Picker
            Button(
                onClick = { modelPicker.launch(arrayOf("*/*")) },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(8.dp),
                enabled = !state.isCopying && !state.isAutoLoading
            ) {
                Text("Select Model File (.gguf)")
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Validation Card
            if (selectedFileSize != null) {
                ValidationCard(
                    fileSize = selectedFileSize!!,
                    ramInfo = ramInfo,
                    onLoad = { viewModel.confirmAndLoad() },
                    isProcessing = state.isCopying || state.isAutoLoading
                )
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Section 4: ARM Tip
            ArmTip()
        }

        // Loading Overlay
        if (state.isCopying || state.isAutoLoading) {
            Surface(
                color = Color.Black.copy(alpha = 0.5f),
                modifier = Modifier.fillMaxSize()
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator(color = Color.White)
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Loading model... this may take 30-60 seconds",
                        color = Color.White,
                        style = MaterialTheme.typography.bodyLarge
                    )
                }
            }
        }
    }

    if (showBottomSheet && ramInfo != null) {
        ModalBottomSheet(
            onDismissRequest = { showBottomSheet = false },
            sheetState = sheetState
        ) {
            RecommendationSheetContent(
                ramInfo = ramInfo!!,
                onRefresh = { viewModel.refreshRamInfo(context) },
                onDismiss = { showBottomSheet = false }
            )
        }
    }
}

@Composable
fun RamStatusCard(ramInfo: RamInfo?, onRefresh: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("RAM Status", style = MaterialTheme.typography.titleMedium)
                IconButton(onClick = onRefresh) {
                    Icon(Icons.Default.Refresh, contentDescription = "Refresh")
                }
            }

            if (ramInfo != null) {
                val availableGB = ramInfo.availableBytes.toFloat() / (1024 * 1024 * 1024)
                val totalGB = ramInfo.totalBytes.toFloat() / (1024 * 1024 * 1024)
                val usedRatio = 1f - (ramInfo.availableBytes.toFloat() / ramInfo.totalBytes)

                Text("Available RAM: ${"%.1f".format(availableGB)} GB")
                Text("Total RAM: ${"%.1f".format(totalGB)} GB", style = MaterialTheme.typography.bodySmall)

                Spacer(modifier = Modifier.height(8.dp))

                val progressColor = when {
                    availableGB > 4.0f -> Color(0xFF4CAF50)
                    availableGB > 2.5f -> Color(0xFFFFC107)
                    else -> Color(0xFFF44336)
                }

                LinearProgressIndicator(
                    progress = usedRatio,
                    modifier = Modifier.fillMaxWidth().height(8.dp),
                    color = progressColor,
                    trackColor = progressColor.copy(alpha = 0.2f)
                )

                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Close background apps for more headroom before loading",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline
                )
            } else {
                CircularProgressIndicator(modifier = Modifier.size(24.dp))
            }
        }
    }
}

@Composable
fun ValidationCard(fileSize: Long, ramInfo: RamInfo?, onLoad: () -> Unit, isProcessing: Boolean) {
    val fileSizeGB = fileSize.toFloat() / (1024 * 1024 * 1024)
    val maxBytes = ramInfo?.recommendedMaxBytes ?: 0L
    
    val (status, message, color) = when {
        fileSize < maxBytes -> Triple("Fits comfortably", "✅ Fits comfortably", Color(0xFF4CAF50))
        fileSize < maxBytes * 1.2f -> Triple("Tight fit", "⚠️ Tight fit — close background apps first", Color(0xFFFFC107))
        else -> Triple("Too large", "❌ Too large for available RAM — choose a smaller quant", Color(0xFFF44336))
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.1f))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Selected File Info", fontWeight = FontWeight.Bold)
            Text("Size: ${"%.2f".format(fileSizeGB)} GB")
            Spacer(modifier = Modifier.height(8.dp))
            Text(message, color = color, fontWeight = FontWeight.Medium)
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Button(
                onClick = onLoad,
                modifier = Modifier.fillMaxWidth(),
                enabled = !isProcessing && (fileSize < maxBytes * 1.2f),
                colors = ButtonDefaults.buttonColors(containerColor = color)
            ) {
                Text("Load Model", color = Color.White)
            }
        }
    }
}

@Composable
fun RecommendationSheetContent(ramInfo: RamInfo, onRefresh: () -> Unit, onDismiss: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
            .padding(bottom = 32.dp)
    ) {
        Text("Recommendations", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))
        
        Text(
            text = "Recommended model size: under ${"%.1f".format(ramInfo.recommendedMaxGB)} GB",
            fontWeight = FontWeight.Bold
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        Text("Preferred Quant Types:")
        ramInfo.preferredQuants.forEach { quant ->
            val suffix = if (quant == "IQ4_NL") " (recommended for ARM devices)" else ""
            Text("• $quant$suffix", modifier = Modifier.padding(start = 8.dp).padding(vertical = 2.dp))
        }

        if (ramInfo.warning != null) {
            Spacer(modifier = Modifier.height(16.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Warning, contentDescription = null, tint = Color(0xFFFFC107))
                Spacer(modifier = Modifier.width(8.dp))
                Text(ramInfo.warning, color = Color(0xFFFFC107), fontWeight = FontWeight.Medium)
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
        
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            TextButton(onClick = onRefresh) { Text("Refresh") }
            Spacer(modifier = Modifier.width(8.dp))
            Button(onClick = onDismiss) { Text("Got it") }
        }
    }
}

@Composable
fun ArmTip() {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f),
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.Top) {
            Text("💡", fontSize = 20.sp)
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = "IQ4_NL quants are recommended for Android devices — they use ARM NEON optimizations for faster prompt processing at the same file size as Q4_K_S.",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}
