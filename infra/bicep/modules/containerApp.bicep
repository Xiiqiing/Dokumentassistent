// ---------------------------------------------------------------------------
// Generic Container App module.
// Used for qdrant, api, ui — they differ only in image, port, env,
// secrets, command, and probes.
// ---------------------------------------------------------------------------

@description('Container App resource name')
param name string

@description('Azure region')
param location string

@description('Parent ACA environment ID')
param containerAppEnvId string

@description('Container name inside the app')
param containerName string

@description('Container image with tag')
param image string

@description('vCPU as string (Bicep json() helper expects string)')
param cpu string = '0.5'

@description('Memory allocation, e.g. 1Gi')
param memory string = '1Gi'

@description('Container port for ingress')
param targetPort int

@description('Whether ingress is exposed publicly')
param externalIngress bool = false

param minReplicas int = 1
param maxReplicas int = 1

@description('Optional entrypoint override')
param command array = []

@description('Environment variables (each item: { name, value? , secretRef? })')
param envVars array = []

@description('Secrets exposed to the container')
@secure()
param secrets array = []

@description('Private registry credentials list')
param registries array = []

@description('Optional readiness probe object (ACA probe schema)')
param readinessProbe object = {}

@description('Optional liveness probe object (ACA probe schema)')
param livenessProbe object = {}

@description('Volume mounts: [{ volumeName, mountPath }]')
param volumeMounts array = []

@description('Volumes: [{ name, storageName, storageType }]')
param volumes array = []

var probes = union(
  empty(readinessProbe) ? [] : [readinessProbe],
  empty(livenessProbe) ? [] : [livenessProbe]
)

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  properties: {
    managedEnvironmentId: containerAppEnvId
    configuration: {
      ingress: {
        external: externalIngress
        targetPort: targetPort
        transport: 'http'
      }
      registries: registries
      secrets: secrets
    }
    template: {
      containers: [
        {
          name: containerName
          image: image
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: envVars
          command: empty(command) ? null : command
          probes: probes
          volumeMounts: volumeMounts
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
      }
      volumes: volumes
    }
  }
}

output id string = app.id
output name string = app.name
output fqdn string = app.properties.configuration.ingress.fqdn
