// ---------------------------------------------------------------------------
// KU Doc Assistant - Azure Container Apps deployment
// Deploys: Container App Environment, ACR, Qdrant, API, and UI containers
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

@description('Google API key (required when llmProvider is google_genai)')
@secure()
param googleApiKey string = ''

@description('OpenAI API key (required when llmProvider is openai)')
@secure()
param openaiApiKey string = ''

@description('Azure OpenAI API key')
@secure()
param azureOpenaiApiKey string = ''

@description('Azure OpenAI endpoint')
param azureOpenaiEndpoint string = ''

@description('Azure OpenAI deployment name')
param azureOpenaiDeployment string = ''

// ---------------------------------------------------------------------------
// Log Analytics + Container App Environment
// ---------------------------------------------------------------------------

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${baseName}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${baseName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Azure Container Registry
// ---------------------------------------------------------------------------

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: replace('${baseName}acr', '-', '')
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

// ---------------------------------------------------------------------------
// Shared storage for Qdrant persistence
// ---------------------------------------------------------------------------

resource qdrantStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: containerAppEnv
  name: 'qdrantstorage'
  properties: {
    azureFile: {
      accountName: storageAccount.name
      accountKey: storageAccount.listKeys().keys[0].value
      shareName: fileShare.name
      accessMode: 'ReadWrite'
    }
  }
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: replace('${baseName}stor', '-', '')
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
}

resource fileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: 'qdrant-data'
  properties: { shareQuota: 5 }
}

// ---------------------------------------------------------------------------
// Qdrant Container App
// ---------------------------------------------------------------------------

resource qdrantApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${baseName}-qdrant'
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: false
        targetPort: 6333
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'qdrant'
          image: 'qdrant/qdrant:latest'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          volumeMounts: [
            { volumeName: 'qdrant-vol', mountPath: '/qdrant/storage' }
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 1 }
      volumes: [
        {
          name: 'qdrant-vol'
          storageName: qdrantStorage.name
          storageType: 'AzureFile'
        }
      ]
    }
  }
}

// ---------------------------------------------------------------------------
// API Container App
// ---------------------------------------------------------------------------

resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${baseName}-api'
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'acr-password', value: acr.listCredentials().passwords[0].value }
        { name: 'google-api-key', value: googleApiKey }
        { name: 'openai-api-key', value: openaiApiKey }
        { name: 'azure-openai-api-key', value: azureOpenaiApiKey }
      ]
      ingress: {
        external: false
        targetPort: 8000
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acr.properties.loginServer}/${baseName}-api:${imageTag}'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'QDRANT_URL', value: 'http://${qdrantApp.properties.configuration.ingress.fqdn}' }
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
          probes: [
            {
              type: 'Readiness'
              httpGet: { path: '/health/ready', port: 8000 }
              initialDelaySeconds: 30
              periodSeconds: 15
            }
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 3 }
    }
  }
}

// ---------------------------------------------------------------------------
// UI Container App (Streamlit)
// ---------------------------------------------------------------------------

resource uiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${baseName}-ui'
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'acr-password', value: acr.listCredentials().passwords[0].value }
      ]
      ingress: {
        external: true
        targetPort: 8501
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'ui'
          image: '${acr.properties.loginServer}/${baseName}-ui:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'API_BASE_URL', value: 'http://${apiApp.properties.configuration.ingress.fqdn}' }
          ]
          command: [
            'streamlit', 'run', 'src/ui/app.py'
            '--server.port=8501', '--server.address=0.0.0.0'
            '--server.headless=true', '--browser.gatherUsageStats=false'
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 3 }
    }
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

output uiUrl string = 'https://${uiApp.properties.configuration.ingress.fqdn}'
output apiInternalUrl string = 'http://${apiApp.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acr.properties.loginServer
