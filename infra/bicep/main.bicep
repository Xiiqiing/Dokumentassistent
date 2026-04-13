// ---------------------------------------------------------------------------
// KU Doc Assistant - root composition.
// Wires platform, storage, and containerApp modules together.
// ---------------------------------------------------------------------------

targetScope = 'resourceGroup'

@description('Base name used to derive resource names')
param baseName string = 'kudocassist'

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Container image tag')
param imageTag string = 'latest'

@description('LLM provider (ollama, openai, azure_openai, google_genai, etc.)')
param llmProvider string = 'google_genai'

@description('Embedding provider (local, openai, azure_openai, google_genai, etc.)')
param embeddingProvider string = 'local'

@description('Generation model name')
param generationModel string = 'gemini-2.5-flash'

@secure()
@description('Google API key (required when llmProvider is google_genai)')
param googleApiKey string = ''

@secure()
@description('OpenAI API key (required when llmProvider is openai)')
param openaiApiKey string = ''

@secure()
@description('Azure OpenAI API key')
param azureOpenaiApiKey string = ''

@description('Azure OpenAI endpoint URL')
param azureOpenaiEndpoint string = ''

@description('Azure OpenAI deployment name')
param azureOpenaiDeployment string = ''

// ---------------------------------------------------------------------------
// Platform + storage
// ---------------------------------------------------------------------------

module platform 'modules/platform.bicep' = {
  name: 'platform'
  params: {
    baseName: baseName
    location: location
  }
}

module qdrantStorage 'modules/storage.bicep' = {
  name: 'qdrantStorage'
  params: {
    baseName: baseName
    location: location
    containerAppEnvName: platform.outputs.containerAppEnvName
  }
}

// Existing reference so we can call listCredentials() without re-declaring ACR
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: platform.outputs.acrName
}

var acrRegistry = {
  server: platform.outputs.acrLoginServer
  username: acr.listCredentials().username
  passwordSecretRef: 'acr-password'
}

var acrPasswordSecret = {
  name: 'acr-password'
  value: acr.listCredentials().passwords[0].value
}

// ---------------------------------------------------------------------------
// Qdrant — internal stateful service
// ---------------------------------------------------------------------------

module qdrant 'modules/containerApp.bicep' = {
  name: 'qdrant'
  params: {
    name: '${baseName}-qdrant'
    location: location
    containerAppEnvId: platform.outputs.containerAppEnvId
    containerName: 'qdrant'
    image: 'qdrant/qdrant:latest'
    cpu: '0.5'
    memory: '1Gi'
    targetPort: 6333
    externalIngress: false
    volumeMounts: [
      { volumeName: 'qdrant-vol', mountPath: '/qdrant/storage' }
    ]
    volumes: [
      {
        name: 'qdrant-vol'
        storageName: qdrantStorage.outputs.envStorageName
        storageType: 'AzureFile'
      }
    ]
  }
}

// ---------------------------------------------------------------------------
// API — internal FastAPI backend
// ---------------------------------------------------------------------------

module api 'modules/containerApp.bicep' = {
  name: 'api'
  params: {
    name: '${baseName}-api'
    location: location
    containerAppEnvId: platform.outputs.containerAppEnvId
    containerName: 'api'
    image: '${platform.outputs.acrLoginServer}/${baseName}-api:${imageTag}'
    cpu: '1.0'
    memory: '2Gi'
    targetPort: 8000
    externalIngress: false
    maxReplicas: 3
    registries: [acrRegistry]
    secrets: [
      acrPasswordSecret
      { name: 'google-api-key', value: googleApiKey }
      { name: 'openai-api-key', value: openaiApiKey }
      { name: 'azure-openai-api-key', value: azureOpenaiApiKey }
    ]
    envVars: [
      { name: 'QDRANT_URL', value: 'http://${qdrant.outputs.fqdn}' }
      { name: 'LLM_PROVIDER', value: llmProvider }
      { name: 'EMBEDDING_PROVIDER', value: embeddingProvider }
      { name: 'GENERATION_MODEL', value: generationModel }
      { name: 'GOOGLE_API_KEY', secretRef: 'google-api-key' }
      { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
      { name: 'AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-api-key' }
      { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenaiEndpoint }
      { name: 'AZURE_OPENAI_DEPLOYMENT', value: azureOpenaiDeployment }
      { name: 'LOG_LEVEL', value: 'INFO' }
    ]
    readinessProbe: {
      type: 'Readiness'
      httpGet: { path: '/health/ready', port: 8000 }
      initialDelaySeconds: 30
      periodSeconds: 15
    }
    livenessProbe: {
      type: 'Liveness'
      httpGet: { path: '/health', port: 8000 }
      periodSeconds: 30
    }
  }
}

// ---------------------------------------------------------------------------
// UI — public Streamlit frontend
// ---------------------------------------------------------------------------

module ui 'modules/containerApp.bicep' = {
  name: 'ui'
  params: {
    name: '${baseName}-ui'
    location: location
    containerAppEnvId: platform.outputs.containerAppEnvId
    containerName: 'ui'
    image: '${platform.outputs.acrLoginServer}/${baseName}-ui:${imageTag}'
    cpu: '0.5'
    memory: '1Gi'
    targetPort: 8501
    externalIngress: true
    maxReplicas: 3
    registries: [acrRegistry]
    secrets: [acrPasswordSecret]
    command: [
      'streamlit', 'run', 'src/ui/app.py'
      '--server.port=8501', '--server.address=0.0.0.0'
      '--server.headless=true', '--browser.gatherUsageStats=false'
    ]
    envVars: [
      { name: 'API_BASE_URL', value: 'http://${api.outputs.fqdn}' }
    ]
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

output uiUrl string = 'https://${ui.outputs.fqdn}'
output apiInternalUrl string = 'http://${api.outputs.fqdn}'
output acrLoginServer string = platform.outputs.acrLoginServer
