from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("MultiToolAssistant")

# ============================================================
# CALCULATOR
# ============================================================

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b."""

    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b

# ============================================================
# WEATHER
# ============================================================

import requests


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather information for a city."""

    # =====================================================
    # 1. INPUT VALIDATION
    # =====================================================

    if not city or not city.strip():
        return "Error: Please provide a valid city name."

    city = city.strip()


    try:

        # =================================================
        # 2. GEOCODING API
        # =================================================

        geocoding_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        geocoding_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )

        # HTTP errors such as 400, 404, 500
        geo_response.raise_for_status()

        geo_data = geo_response.json()


        # =================================================
        # 3. CITY NOT FOUND
        # =================================================

        if not geo_data.get("results"):
            return (
                f"Error: City '{city}' "
                "was not found."
            )


        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        city_name = location["name"]
        country = location.get("country", "Unknown")


        # =================================================
        # 4. WEATHER API
        # =================================================

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "wind_speed_10m,"
                "weather_code"
            ),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()


        # =================================================
        # 5. VALIDATE API RESPONSE
        # =================================================

        if "current" not in weather_data:
            return (
                "Error: Weather API returned "
                "an unexpected response."
            )


        current = weather_data["current"]


        # =================================================
        # 6. EXTRACT DATA
        # =================================================

        temperature = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind_speed = current.get("wind_speed_10m")
        weather_code = current.get("weather_code")


        # =================================================
        # 7. WEATHER CODE
        # =================================================

        weather_conditions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
        }

        condition = weather_conditions.get(
            weather_code,
            "Unknown weather condition",
        )


        # =================================================
        # 8. FINAL RESULT
        # =================================================

        return (
            f"Weather in {city_name}, {country}:\n"
            f"Condition: {condition}\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind speed: {wind_speed} km/h"
        )


    # =====================================================
    # 9. NETWORK / HTTP ERRORS
    # =====================================================

    except requests.Timeout:
        return (
            "Error: Weather API request timed out. "
            "Please try again later."
        )


    except requests.ConnectionError:
        return (
            "Error: Could not connect to the weather API. "
            "Please check the network connection."
        )


    except requests.HTTPError as e:
        return (
            f"Error: Weather API returned an HTTP error: {e}"
        )


    except requests.RequestException as e:
        return (
            f"Error: Weather API request failed: {e}"
        )


    # =====================================================
    # 10. JSON / DATA ERRORS
    # =====================================================

    except ValueError:
        return (
            "Error: Weather API returned invalid JSON data."
        )


    # =====================================================
    # 11. UNEXPECTED ERROR
    # =====================================================

    except Exception as e:
        return (
            f"Error: Unexpected server error: {e}"
        )

# ============================================================
# TEXT
# ============================================================

@mcp.tool()
def word_count(text: str) -> int:
    """Count the number of words in a text."""

    return len(text.split())

# ============================================================
# RUN MCP SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()