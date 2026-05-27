"""
FastAPI 入口 — 企业知识管理系统 REST API

提供三组接口:
  1. /api/ingest   — 文档上传 & 入库
  2. /api/qa       — 智能问答
  3. /api/admin    — 管理（统计、更新触发）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import os
import shutil
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.knowledge_graph import KnowledgeGraphService
from services.memory_service import MemoryService
from services.vector_store import VectorStoreService

vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
memory_service = MemoryService()
doc_parser = DocParserAgent()
extractor = KnowledgeExtractAgent()
workflows: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)
    try:
        await vector_store.init()
    except Exception:
        pass
    try:
        await knowledge_graph.init()
    except Exception:
        pass
    workflows.update(
        build_knowledge_graph_workflow(
            vector_store=vector_store, 
            knowledge_graph=knowledge_graph,
            memory_service=memory_service
        )
    )
    yield
    await knowledge_graph.close()


app = FastAPI(
    title="AgentKnowledgeHub — 多Agent企业知识管理系统",
    description="支持多模态RAG、知识图谱、增量更新的企业级知识管理 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None
    user_id: str | None = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]


class UpdateRequest(BaseModel):
    file_path: str
    change_type: str = "modified"


class UpdateResponse(BaseModel):
    file_path: str
    vectors_added: int
    vectors_deleted: int
    entities_added: int
    relations_added: int
    success: bool
    processing_time_ms: float


# ── Ingest Endpoints ─────────────────────────────────────────

@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(file: UploadFile = File(...)):
    """上传并解析文档，自动入库到向量库和知识图谱"""
    import logging
    logger = logging.getLogger(__name__)
    
    save_path = os.path.join(settings.upload_dir, file.filename or "unknown")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        logger.info(f"开始解析文档: {save_path}")
        chunks = await doc_parser.parse_batch([save_path])
        logger.info(f"文档解析完成，共 {len(chunks)} 个块")
        
        logger.info("开始知识提取")
        extractions = await extractor.extract(chunks)
        logger.info(f"知识提取完成，共 {len(extractions)} 个提取结果")
        
        if vector_store:
            logger.info("开始存储向量")
            await vector_store.add_chunks(chunks)
            logger.info("向量存储完成")
        
        if knowledge_graph:
            logger.info("开始存储知识图谱")
            for ext in extractions:
                for ent in ext.entities:
                    await knowledge_graph.upsert_entity(ent)
                for rel in ext.relations:
                    await knowledge_graph.add_relation(rel)
            logger.info("知识图谱存储完成")

        total_entities = sum(len(e.entities) for e in extractions)
        total_relations = sum(len(e.relations) for e in extractions)

        return IngestResponse(
            file_name=file.filename or "unknown",
            chunks_count=len(chunks),
            entities_count=total_entities,
            relations_count=total_relations,
            status="success",
        )
    except Exception as e:
        logger.error(f"上传失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["文档入库"])
async def upload_batch(files: list[UploadFile] = File(...)):
    """批量上传文档"""
    results = []
    for file in files:
        resp = await upload_document(file)
        results.append(resp)
    return results


@app.get("/api/ingest/documents", tags=["文档入库"])
async def get_documents():
    """获取已上传文档列表"""
    documents = []
    if os.path.exists(settings.upload_dir):
        for filename in os.listdir(settings.upload_dir):
            filepath = os.path.join(settings.upload_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                documents.append({
                    "id": filename,
                    "name": filename,
                    "size": stat.st_size,
                    "upload_time": stat.st_ctime,
                    "chunks_count": 0
                })
    return documents


@app.delete("/api/ingest/documents/{file_name}", tags=["文档入库"])
async def delete_document(file_name: str):
    """删除已上传的文档"""
    filepath = os.path.join(settings.upload_dir, file_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        os.remove(filepath)
        return {"success": True, "message": f"Document {file_name} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# ── QA Endpoints ─────────────────────────────────────────────

@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(req: QuestionRequest):
    """智能问答 — 混合检索 + 知识图谱推理 + 记忆系统"""
    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    inputs = {"question": req.question}
    if req.session_id:
        inputs["session_id"] = req.session_id
    if req.user_id:
        inputs["user_id"] = req.user_id

    result = await qa_wf.ainvoke(inputs)
    qa_result = result.get("result")
    if not qa_result:
        raise HTTPException(status_code=500, detail="QA failed")

    return QuestionResponse(
        question=qa_result.question,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        sources=[
            {"content": c.content[:200], "source": c.source, "score": c.score, "type": c.retrieval_type}
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
    )


# ── Admin Endpoints ──────────────────────────────────────────

@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats():
    """获取系统统计信息"""
    vs_stats = await vector_store.get_stats()
    kg_stats = await knowledge_graph.get_stats()
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats)


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["系统管理"])
async def trigger_update(req: UpdateRequest):
    """手动触发知识更新"""
    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="Update workflow not initialized")

    change = DocumentChange(
        file_path=req.file_path,
        change_type=ChangeType(req.change_type),
    )
    result = await update_wf.ainvoke({"changes": [change]})
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="Update failed")

    r = results[0]
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


# ── Memory Endpoints ─────────────────────────────────────────

class UserProfileRequest(BaseModel):
    user_id: str
    name: str | None = None
    background: str | None = None
    preferences: dict | None = None

class UserProfileResponse(BaseModel):
    user_id: str
    name: str
    background: str
    preferences: dict
    interaction_count: int
    last_active: str
    created_at: str

class PersonalityRequest(BaseModel):
    user_id: str
    warmth: float | None = None
    expertise: float | None = None
    humor: float | None = None
    empathy: float | None = None

class PersonalityResponse(BaseModel):
    user_id: str
    warmth: float
    expertise: float
    humor: float
    empathy: float

class MemoryRetrieveRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    top_k: int = 5

class MemoryEventResponse(BaseModel):
    session_id: str
    timestamp: str
    user_input: str
    agent_response: str
    summary: str
    topics: list[str]
    importance: float

# ── Conversation Models ───────────────────────────────────────

class MessageRequest(BaseModel):
    role: str  # user | assistant | system
    content: str
    message_id: str | None = None
    metadata: dict | None = None

class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    metadata: dict

class ConversationRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    title: str | None = None
    metadata: dict | None = None

class ConversationResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    messages: list[MessageResponse]
    created_at: str
    updated_at: str
    metadata: dict

class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int

class ConversationSearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    limit: int = 20

class DeleteMessageRequest(BaseModel):
    message_id: str

@app.get("/api/memory/profile/{user_id}", response_model=UserProfileResponse, tags=["记忆管理"])
async def get_user_profile(user_id: str):
    """获取用户画像"""
    profile = await memory_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    return UserProfileResponse(
        user_id=profile.user_id,
        name=profile.name,
        background=profile.background,
        preferences=profile.preferences,
        interaction_count=profile.interaction_count,
        last_active=profile.last_active,
        created_at=profile.created_at,
    )

@app.post("/api/memory/profile", response_model=UserProfileResponse, tags=["记忆管理"])
async def update_user_profile(req: UserProfileRequest):
    """更新用户画像"""
    profile = await memory_service.get_user_profile(req.user_id)
    if not profile:
        from services.memory_models import UserProfile
        profile = UserProfile(user_id=req.user_id)
    
    if req.name is not None:
        profile.name = req.name
    if req.background is not None:
        profile.background = req.background
    if req.preferences is not None:
        profile.preferences.update(req.preferences)
    
    await memory_service.update_user_profile(req.user_id, profile)
    return UserProfileResponse(
        user_id=profile.user_id,
        name=profile.name,
        background=profile.background,
        preferences=profile.preferences,
        interaction_count=profile.interaction_count,
        last_active=profile.last_active,
        created_at=profile.created_at,
    )

@app.get("/api/memory/personality/{user_id}", response_model=PersonalityResponse, tags=["记忆管理"])
async def get_personality(user_id: str):
    """获取AI个性参数"""
    personality = await memory_service.get_personality(user_id)
    return PersonalityResponse(
        user_id=user_id,
        warmth=personality.warmth,
        expertise=personality.expertise,
        humor=personality.humor,
        empathy=personality.empathy,
    )

@app.post("/api/memory/personality", response_model=PersonalityResponse, tags=["记忆管理"])
async def update_personality(req: PersonalityRequest):
    """更新AI个性参数"""
    personality = await memory_service.get_personality(req.user_id)
    
    if req.warmth is not None:
        personality.warmth = max(0, min(100, req.warmth))
    if req.expertise is not None:
        personality.expertise = max(0, min(100, req.expertise))
    if req.humor is not None:
        personality.humor = max(0, min(100, req.humor))
    if req.empathy is not None:
        personality.empathy = max(0, min(100, req.empathy))
    
    await memory_service.update_personality(req.user_id, personality)
    return PersonalityResponse(
        user_id=req.user_id,
        warmth=personality.warmth,
        expertise=personality.expertise,
        humor=personality.humor,
        empathy=personality.empathy,
    )

@app.post("/api/memory/retrieve", response_model=list[MemoryEventResponse], tags=["记忆管理"])
async def retrieve_memory(req: MemoryRetrieveRequest):
    """检索相关历史记忆"""
    events = await memory_service.retrieve_long_term(req.query, req.top_k)
    return [MemoryEventResponse(
        session_id=e.session_id,
        timestamp=e.timestamp,
        user_input=e.user_input,
        agent_response=e.agent_response,
        summary=e.summary,
        topics=e.topics,
        importance=e.importance,
    ) for e in events]

# ── Conversation Endpoints ────────────────────────────────────

@app.post("/api/conversations", response_model=ConversationResponse, tags=["对话管理"])
async def create_conversation(req: ConversationRequest):
    """创建新对话会话"""
    conversation = await memory_service.create_conversation(
        session_id=req.session_id,
        user_id=req.user_id or "",
        title=req.title or "",
        metadata=req.metadata
    )
    return ConversationResponse(
        session_id=conversation.session_id,
        user_id=conversation.user_id,
        title=conversation.title,
        messages=[],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        metadata=conversation.metadata
    )

@app.get("/api/conversations", response_model=list[ConversationResponse], tags=["对话管理"])
async def list_conversations(user_id: str = "", limit: int = 20, offset: int = 0):
    """获取所有历史对话列表"""
    conversations = await memory_service.list_conversations(user_id=user_id, limit=limit, offset=offset)
    return [ConversationResponse(
        session_id=c.session_id,
        user_id=c.user_id,
        title=c.title,
        messages=[],
        created_at=c.created_at,
        updated_at=c.updated_at,
        metadata=c.metadata
    ) for c in conversations]

@app.post("/api/conversations/search", response_model=list[ConversationResponse], tags=["对话管理"])
async def search_conversations(req: ConversationSearchRequest):
    """搜索对话（按标题或消息内容）"""
    conversations = await memory_service.search_conversations(
        query=req.query,
        user_id=req.user_id or "",
        limit=req.limit
    )
    return [ConversationResponse(
        session_id=c.session_id,
        user_id=c.user_id,
        title=c.title,
        messages=[],
        created_at=c.created_at,
        updated_at=c.updated_at,
        metadata=c.metadata
    ) for c in conversations]

@app.get("/api/conversations/{session_id}", response_model=ConversationResponse, tags=["对话管理"])
async def get_conversation(session_id: str):
    """获取单个会话详情（继续对话）"""
    conversation = await memory_service.get_conversation(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        session_id=conversation.session_id,
        user_id=conversation.user_id,
        title=conversation.title,
        messages=[MessageResponse(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            timestamp=m.timestamp,
            metadata=m.metadata
        ) for m in conversation.messages],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        metadata=conversation.metadata
    )

@app.post("/api/conversations/{session_id}/messages", response_model=MessageResponse, tags=["对话管理"])
async def add_message(session_id: str, req: MessageRequest):
    """添加消息到会话（存储对话）"""
    message = await memory_service.save_message(
        session_id=session_id,
        role=req.role,
        content=req.content,
        message_id=req.message_id or "",
        metadata=req.metadata
    )
    return MessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        timestamp=message.timestamp,
        metadata=message.metadata
    )

@app.put("/api/conversations/{session_id}/title", tags=["对话管理"])
async def update_conversation_title(session_id: str, title: str):
    """更新会话标题"""
    success = await memory_service.update_conversation_title(session_id, title)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True, "message": "Title updated"}

@app.delete("/api/conversations/{session_id}", tags=["对话管理"])
async def delete_conversation(session_id: str):
    """删除整个会话"""
    success = await memory_service.delete_conversation(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True, "message": "Conversation deleted"}

@app.delete("/api/conversations/{session_id}/messages/{message_id}", tags=["对话管理"])
async def delete_message(session_id: str, message_id: str):
    """删除单条消息"""
    success = await memory_service.delete_message(session_id, message_id)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"success": True, "message": "Message deleted"}

@app.get("/api/health", tags=["系统管理"])
async def health():
    return {"status": "ok", "service": "AgentKnowledgeHub"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
