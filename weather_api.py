from mcp.server.fastmcp import FastMCP


mcp=FastMCP('Weather')


@mcp.tool()
def get_weather(city: str) -> str:
    """
    Fetches the current weather for a given city using a weather API.
    
    Args:
        city (str): The name of the city to fetch the weather for.
    """
    return "The current weather is rainy"

if __name__ == "__main__":
    mcp.run(transport="stdio")

