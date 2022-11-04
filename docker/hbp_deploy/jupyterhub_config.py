# keycloak integration
import os
from oauthenticator.generic import GenericOAuthenticator
c.JupyterHub.authenticator_class = GenericOAuthenticator
c.GenericOAuthenticator.login_service = 'keycloak'

c.OAuthenticator.client_id= "tvb-multiscale"
c.OAuthenticator.scope = ["profile"]
c.OAuthenticator.client_secret= os.environ['KEYCLOAK_CLIENT_SECRET']
c.GenericOAuthenticator.token_url= "https://iam.humanbrainproject.eu/auth/realms/hbp/protocol/openid-connect/token"
c.GenericOAuthenticator.userdata_url= "https://iam.humanbrainproject.eu/auth/realms/hbp/protocol/openid-connect/userinfo"
c.GenericOAuthenticator.userdata_method= "GET"
c.GenericOAuthenticator.userdata_params= {'state': 'state'}
c.GenericOAuthenticator.username_key= "preferred_username"

# persistent storage for the notebooks
c.KubeSpawner.user_storage_pvc_ensure = True

c.KubeSpawner.pvc_name_template = 'pvc-{username}'
c.KubeSpawner.user_storage_capacity = '1Gi'
c.KubeSpawner.storage_class = 'managed-nfs-storage'

c.KubeSpawner.volumes = [
    {
        'name': 'data',
        'persistentVolumeClaim': {
            'claimName': c.KubeSpawner.pvc_name_template
        }
    }
]

c.KubeSpawner.volume_mounts = [
    {
        'name': 'data',
        'mountPath': '/home/jovyan/packages/notebooks/Contributed-Notebooks'
    }
]