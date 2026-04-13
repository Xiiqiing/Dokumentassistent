output "ui_url" {
  description = "Public URL of the Streamlit UI"
  value       = "https://${module.ui.fqdn}"
}

output "api_internal_url" {
  description = "Internal URL of the FastAPI backend"
  value       = "http://${module.api.fqdn}"
}

output "acr_login_server" {
  description = "Azure Container Registry login server"
  value       = module.platform.acr_login_server
}
