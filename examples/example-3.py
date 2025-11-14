async def fetch_user_data(user_id, session, timeout=30):
    url = f"https://api.example.com/users/{user_id}"
    async with session.get(url, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"Failed to fetch user {user_id}")
        return await response.json()