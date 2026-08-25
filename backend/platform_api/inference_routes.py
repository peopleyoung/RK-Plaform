from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials

from .auth import bearer, get_context, require_admin
from .context import AppContext
from .contracts import (
    AgentDesiredState,
    DeploymentCreate,
    DeploymentEventResponse,
    DeploymentListResponse,
    DeploymentResponse,
    DeploymentTargetReport,
    DeploymentTargetResponse,
    InferenceGraphRevisionResponse,
    InferenceGraphTaskCreate,
    InferenceNodeCreate,
    InferenceNodeCreated,
    InferenceNodeHeartbeat,
    InferenceNodeListResponse,
    InferenceNodeRegistration,
    InferenceNodeRegistrationResponse,
    InferenceNodeResponse,
    InferenceSummaryResponse,
    InferenceTaskListResponse,
    InferenceTaskResponse,
    InferenceTaskUpdate,
    ModelReleaseCreate,
    ModelReleaseListResponse,
    ModelReleaseResponse,
    NodeGroupCreate,
    NodeGroupResponse,
    NodeGroupUpdate,
)
from .errors import AuthenticationError
from .inference_graph import (
    GraphValidationRequest,
    GraphValidationResponse,
    OperatorCatalogResponse,
    catalog_response,
)
from .inference_service import InferenceService, model_release_response

router = APIRouter()
Admin = Annotated[None, Depends(require_admin)]
Context = Annotated[AppContext, Depends(get_context)]
Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]


def _access_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError()
    return credentials.credentials


@router.get("/inference-operator-catalog", response_model=OperatorCatalogResponse)
def get_inference_operator_catalog(_: Admin) -> OperatorCatalogResponse:
    return catalog_response()


@router.post("/inference-graphs/validate", response_model=GraphValidationResponse)
def validate_inference_graph(
    payload: GraphValidationRequest, _: Admin, context: Context
) -> GraphValidationResponse:
    return InferenceService(context).validate_graph(payload)


@router.post(
    "/model-releases",
    response_model=ModelReleaseResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_model_release(
    payload: ModelReleaseCreate, _: Admin, context: Context
) -> ModelReleaseResponse:
    return InferenceService(context).create_release(payload)


@router.get("/model-releases", response_model=ModelReleaseListResponse)
def list_model_releases(
    _: Admin,
    context: Context,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
) -> ModelReleaseListResponse:
    return InferenceService(context).list_releases(page, page_size)


@router.get("/model-releases/{release_id}", response_model=ModelReleaseResponse)
def get_model_release(release_id: str, _: Admin, context: Context) -> ModelReleaseResponse:
    return model_release_response(InferenceService(context).get_release(release_id))


@router.post("/model-releases/{release_id}/publish", response_model=ModelReleaseResponse)
def publish_model_release(release_id: str, _: Admin, context: Context) -> ModelReleaseResponse:
    return InferenceService(context).publish_release(release_id)


@router.post("/model-releases/{release_id}/deprecate", response_model=ModelReleaseResponse)
def deprecate_model_release(release_id: str, _: Admin, context: Context) -> ModelReleaseResponse:
    return InferenceService(context).deprecate_release(release_id)


@router.delete("/model-releases/{release_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model_release(release_id: str, _: Admin, context: Context) -> Response:
    InferenceService(context).delete_release(release_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/node-groups", response_model=NodeGroupResponse, status_code=status.HTTP_201_CREATED)
def create_node_group(payload: NodeGroupCreate, _: Admin, context: Context) -> NodeGroupResponse:
    return InferenceService(context).create_node_group(payload)


@router.get("/node-groups", response_model=list[NodeGroupResponse])
def list_node_groups(_: Admin, context: Context) -> list[NodeGroupResponse]:
    return InferenceService(context).list_node_groups()


@router.put("/node-groups/{group_id}", response_model=NodeGroupResponse)
def update_node_group(
    group_id: str, payload: NodeGroupUpdate, _: Admin, context: Context
) -> NodeGroupResponse:
    return InferenceService(context).update_node_group(group_id, payload)


@router.delete("/node-groups/{group_id}", response_model=NodeGroupResponse)
def delete_node_group(group_id: str, _: Admin, context: Context) -> NodeGroupResponse:
    return InferenceService(context).delete_node_group(group_id)


@router.post(
    "/inference-nodes",
    response_model=InferenceNodeCreated,
    status_code=status.HTTP_201_CREATED,
)
def create_inference_node(
    payload: InferenceNodeCreate, _: Admin, context: Context
) -> InferenceNodeCreated:
    return InferenceService(context).create_node(payload)


@router.get("/inference-nodes", response_model=InferenceNodeListResponse)
def list_inference_nodes(
    _: Admin,
    context: Context,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
) -> InferenceNodeListResponse:
    return InferenceService(context).list_nodes(page, page_size)


@router.get("/inference-summary", response_model=InferenceSummaryResponse)
def get_inference_summary(_: Admin, context: Context) -> InferenceSummaryResponse:
    return InferenceService(context).summary()


@router.post("/inference-nodes/{node_id}/approve", response_model=InferenceNodeResponse)
def approve_inference_node(node_id: str, _: Admin, context: Context) -> InferenceNodeResponse:
    return InferenceService(context).approve_node(node_id)


@router.delete("/inference-nodes/{node_id}", response_model=InferenceNodeResponse)
def retire_inference_node(node_id: str, _: Admin, context: Context) -> InferenceNodeResponse:
    return InferenceService(context).retire_node(node_id)


@router.delete(
    "/inference-nodes/{node_id}/record",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_retired_inference_node(node_id: str, _: Admin, context: Context) -> Response:
    InferenceService(context).delete_retired_node(node_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/inference-agent/register",
    response_model=InferenceNodeRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_inference_node(
    payload: InferenceNodeRegistration, context: Context
) -> InferenceNodeRegistrationResponse:
    return InferenceService(context).register_node(payload)


@router.post("/inference-agent/nodes/{node_id}/heartbeat", response_model=InferenceNodeResponse)
def inference_node_heartbeat(
    node_id: str,
    payload: InferenceNodeHeartbeat,
    credentials: Credentials,
    context: Context,
) -> InferenceNodeResponse:
    return InferenceService(context).heartbeat_node(node_id, _access_token(credentials), payload)


@router.get("/inference-agent/nodes/{node_id}/desired", response_model=AgentDesiredState)
def inference_node_desired_state(
    node_id: str, credentials: Credentials, context: Context
) -> AgentDesiredState:
    return InferenceService(context).desired_state(node_id, _access_token(credentials))


@router.post(
    "/inference-agent/nodes/{node_id}/targets/{target_id}/status",
    response_model=DeploymentTargetResponse,
)
def report_deployment_target(
    node_id: str,
    target_id: str,
    payload: DeploymentTargetReport,
    credentials: Credentials,
    context: Context,
) -> DeploymentTargetResponse:
    return InferenceService(context).report_target(
        node_id, target_id, _access_token(credentials), payload
    )


@router.get("/inference-agent/nodes/{node_id}/artifacts/{artifact_id}/download")
def download_inference_artifact(
    node_id: str, artifact_id: str, credentials: Credentials, context: Context
) -> FileResponse:
    artifact = InferenceService(context).artifact_for_node(
        node_id, artifact_id, _access_token(credentials)
    )
    path = context.storage.require(artifact.storage_key)
    return FileResponse(path, filename=artifact.filename, media_type=artifact.media_type)


@router.post(
    "/inference-tasks",
    response_model=InferenceTaskResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_inference_task(
    payload: InferenceGraphTaskCreate, _: Admin, context: Context
) -> InferenceTaskResponse:
    return InferenceService(context).create_task(payload)


@router.get("/inference-tasks", response_model=InferenceTaskListResponse)
def list_inference_tasks(
    _: Admin,
    context: Context,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
) -> InferenceTaskListResponse:
    return InferenceService(context).list_tasks(page, page_size)


@router.get(
    "/inference-tasks/{task_id}/graph-revisions",
    response_model=list[InferenceGraphRevisionResponse],
)
def list_inference_graph_revisions(
    task_id: str, _: Admin, context: Context
) -> list[InferenceGraphRevisionResponse]:
    return InferenceService(context).list_graph_revisions(task_id)


@router.put("/inference-tasks/{task_id}", response_model=InferenceTaskResponse)
def update_inference_task(
    task_id: str, payload: InferenceTaskUpdate, _: Admin, context: Context
) -> InferenceTaskResponse:
    return InferenceService(context).update_task(task_id, payload)


@router.post("/inference-tasks/{task_id}/stop", response_model=InferenceTaskResponse)
def stop_inference_task(task_id: str, _: Admin, context: Context) -> InferenceTaskResponse:
    return InferenceService(context).stop_task(task_id)


@router.post(
    "/inference-tasks/{task_id}/restart",
    response_model=InferenceTaskResponse,
)
def restart_inference_task(task_id: str, _: Admin, context: Context) -> InferenceTaskResponse:
    return InferenceService(context).restart_task(task_id)


@router.delete("/inference-tasks/{task_id}", response_model=InferenceTaskResponse)
def retire_inference_task(task_id: str, _: Admin, context: Context) -> InferenceTaskResponse:
    return InferenceService(context).retire_task(task_id)


@router.post(
    "/deployments",
    response_model=DeploymentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_deployment(payload: DeploymentCreate, _: Admin, context: Context) -> DeploymentResponse:
    return InferenceService(context).create_deployment(payload)


@router.get("/deployments", response_model=DeploymentListResponse)
def list_deployments(
    _: Admin,
    context: Context,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
) -> DeploymentListResponse:
    return InferenceService(context).list_deployments(page, page_size)


@router.get("/deployments/{deployment_id}", response_model=DeploymentResponse)
def get_deployment(deployment_id: str, _: Admin, context: Context) -> DeploymentResponse:
    return InferenceService(context).get_deployment(deployment_id)


@router.delete(
    "/deployments/{deployment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_deployment(deployment_id: str, _: Admin, context: Context) -> Response:
    InferenceService(context).delete_deployment(deployment_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/deployments/{deployment_id}/events",
    response_model=list[DeploymentEventResponse],
)
def get_deployment_events(
    deployment_id: str,
    _: Admin,
    context: Context,
    after_id: Annotated[int, Query(alias="afterId", ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> list[DeploymentEventResponse]:
    return InferenceService(context).deployment_events(deployment_id, after_id, limit)


@router.post("/deployments/{deployment_id}/rollback", response_model=DeploymentResponse)
def rollback_deployment(deployment_id: str, _: Admin, context: Context) -> DeploymentResponse:
    return InferenceService(context).rollback_deployment(deployment_id)


@router.post("/deployments/{deployment_id}/retry", response_model=DeploymentResponse)
def retry_deployment(deployment_id: str, _: Admin, context: Context) -> DeploymentResponse:
    return InferenceService(context).retry_deployment(deployment_id)
