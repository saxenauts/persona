from persona.core.graph_ops import GraphOps

class UserService:
    @staticmethod
    async def create_user(user_id: str, graph_ops: GraphOps):
        # Check if user already exists
        if await graph_ops.user_exists(user_id):
            return {"message": f"User {user_id} already exists", "status": "exists"}
        
        await graph_ops.create_user(user_id)
        return {"message": f"User {user_id} created successfully", "status": "created"}

    @staticmethod
    async def delete_user(user_id: str, graph_ops: GraphOps):
        await graph_ops.delete_user(user_id)
        return {"message": f"User {user_id} deleted successfully"}