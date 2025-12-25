from neo4j import GraphDatabase
import time

def batch_delete():
    driver = GraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', 'password'))
    
    total_deleted = 0
    batch_size = 5000
    
    while True:
        query = f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(*) as count"
        with driver.session() as session:
            result = session.run(query)
            count = result.single()["count"]
            total_deleted += count
            print(f"Deleted {count} nodes (Total: {total_deleted})")
            if count < batch_size:
                break
    
    driver.close()
    print("Database is now clean.")

if __name__ == '__main__':
    batch_delete()
