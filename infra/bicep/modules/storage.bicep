// ---------------------------------------------------------------------------
// Storage module: storage account, file share, and ACA env binding.
// Used for stateful workloads needing persistent volumes (Qdrant).
// ---------------------------------------------------------------------------

@description('Base name used to derive the storage account name')
param baseName string

@description('Azure region')
param location string

@description('Existing Container App Environment name to bind storage to')
param containerAppEnvName string

@description('File share name')
param shareName string = 'qdrant-data'

@description('ACA environment storage binding name')
param envStorageName string = 'qdrantstorage'

@description('File share quota in GB')
param quotaGb int = 5

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
  name: shareName
  properties: { shareQuota: quotaGb }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: containerAppEnvName
}

resource envStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: containerAppEnv
  name: envStorageName
  properties: {
    azureFile: {
      accountName: storageAccount.name
      accountKey: storageAccount.listKeys().keys[0].value
      shareName: fileShare.name
      accessMode: 'ReadWrite'
    }
  }
}

output envStorageName string = envStorage.name
