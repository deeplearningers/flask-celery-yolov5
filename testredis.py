import redis

if __name__ == "__main__":
    r = redis.Redis(host="localhost",port=6379,decode_responses=True)
    r.set("name","a")
    print(r.get("name"))