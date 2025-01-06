from luna9.core.graph_ops import GraphOps

class UserService:
    @staticmethod
    async def create_user(user_id: str):
        async with GraphOps() as graph_ops:
            await graph_ops.create_user(user_id)
        return {"message": f"User {user_id} created successfully"}

    @staticmethod
    async def delete_user(user_id: str):
        async with GraphOps() as graph_ops:
            await graph_ops.delete_user(user_id)
        return {"message": f"User {user_id} deleted successfully"}