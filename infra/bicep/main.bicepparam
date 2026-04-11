using './main.bicep'

param baseName = 'kudocassist'
param location = 'northeurope'
param imageTag = 'latest'

// LLM configuration -- adjust to your provider
param llmProvider = 'google_genai'
param embeddingProvider = 'local'
param generationModel = 'gemini-2.5-flash'

// Secrets -- pass via CLI: --parameters googleApiKey=<value>
// param googleApiKey = ''
// param openaiApiKey = ''
// param azureOpenaiApiKey = ''
