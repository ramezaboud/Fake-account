"""
Pydantic schemas for the Fake Account Detection API.

This module defines the data models used for request validation
and response serialization in the FastAPI application.
"""

from datetime import datetime
from typing import List, Optional ,Dict

from pydantic import BaseModel, Field, field_validator


class UserProfile(BaseModel):
    """
    Schema for a single user profile.
    
    Attributes:
        name: User display name
        screen_name: User handle/username
        statuses_count: Number of tweets/posts
        followers_count: Number of followers
        friends_count: Number of accounts the user follows
        favourites_count: Number of liked posts
        listed_count: Number of lists the user is on
        created_at: Account creation date
        description: User bio/description
        lang: User language setting
        default_profile: Whether using default profile settings
        verified: Whether account is verified
    """
    name: Optional[str] = Field(None, description="User display name")
    screen_name: str = Field(None,description="User handle/username")
    statuses_count: int = Field(0, ge=0, description="Number of tweets")
    followers_count: int = Field(0, ge=0, description="Number of followers")
    friends_count: int = Field(0, ge=0, description="Number of friends")
    favourites_count: int = Field(0, ge=0, description="Number of favorites")
    listed_count: int = Field(0, ge=0, description="Number of lists")
    created_at: str = Field(..., description="Account creation date. Accepted formats: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, RFC/Twitter date, or ISO8601")
    description: Optional[str] = Field(None, description="User bio")
    lang: Optional[str] = Field("en", description="Language code")
    default_profile: bool = Field(True, description="Using default profile")
    verified: bool = Field(False, description="Is verified")
    
    @field_validator("created_at")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date/time strings and normalize to ISO format.

        Accepts several common formats (date-only, datetime with time, RFC/Twitter
        string, and ISO8601). Returns an ISO-8601 string which downstream
        processing (pandas.to_datetime) will accept.
        """
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%a %b %d %H:%M:%S %z %Y",  # Twitter/RFC style
        ]

        last_err = None
        for fmt in formats:
            try:
                dt = datetime.strptime(v, fmt)
                # Normalize to ISO format (no timezone conversion here)
                return dt.isoformat()
            except Exception as e:
                last_err = e

        # As a final fallback, allow pandas/ISO parsing by attempting datetime.fromisoformat
        try:
            dt = datetime.fromisoformat(v)
            return dt.isoformat()
        except Exception:
            raise ValueError("Invalid date format. Accepted: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS or ISO8601")


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    user: UserProfile = Field(..., description="User profile to analyze")


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    prediction: int = Field(..., description="0=genuine, 1=fake")
    label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    users: List[UserProfile] = Field(..., description="List of user profiles")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction results."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    fake_count: int = Field(..., description="Number of fake accounts detected")
    genuine_count: int = Field(..., description="Number of genuine accounts")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
