from app.clients.knowledge import KnowledgeServiceClient

async def retrieve_knowledge(client: KnowledgeServiceClient, query: str, top_k: int = 8):
    return await client.retrieve(query=query, top_k=top_k)
