package com.brahmadeo.tunedllm

import android.app.ActivityManager
import android.content.Context

data class RamInfo(
    val totalBytes: Long,
    val availableBytes: Long,
    val recommendedMaxBytes: Long, // availableBytes - 800MB headroom
    val recommendedMaxGB: Float,
    val preferredQuants: List<String>,
    val warning: String?
)

fun getDeviceRamInfo(context: Context): RamInfo {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val memoryInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memoryInfo)

    val totalBytes = memoryInfo.totalMem
    val availableBytes = memoryInfo.availMem
    val recommendedMaxBytes = availableBytes - (800 * 1024 * 1024)
    val recommendedMaxGB = recommendedMaxBytes.toFloat() / (1024 * 1024 * 1024)

    val preferredQuants = when {
        recommendedMaxGB >= 5.0f -> listOf("IQ4_NL", "Q4_K_M", "Q5_K_M", "Q5_K_S")
        recommendedMaxGB >= 3.5f -> listOf("IQ4_NL", "Q4_K_M", "Q4_K_S", "IQ4_XS")
        recommendedMaxGB >= 3.0f -> listOf("Q4_K_S", "IQ4_XS", "IQ4_NL", "Q3_K_M")
        recommendedMaxGB >= 2.5f -> listOf("Q3_K_M", "Q3_K_S", "IQ3_M")
        else -> listOf("Q2_K", "IQ2_M")
    }

    val warning = when {
        recommendedMaxGB < 2.5f -> "Very limited RAM — expect slow inference and possible crashes"
        recommendedMaxGB < 3.0f -> "Tight on RAM — close background apps before loading"
        else -> null
    }

    return RamInfo(
        totalBytes = totalBytes,
        availableBytes = availableBytes,
        recommendedMaxBytes = recommendedMaxBytes,
        recommendedMaxGB = recommendedMaxGB,
        preferredQuants = preferredQuants,
        warning = warning
    )
}
