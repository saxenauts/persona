# Luna Memory BYOA Documentation

## Overview

Luna Memory's BYOA (Bring Your Own Algorithm) system allows you to transform static user data into dynamic, business-driving insights. Instead of being limited by predefined schemas and fixed business logic, you can:

- Learn anything about your users through natural language
- Ask complex questions about user behavior and preferences
- Generate new business insights from existing data

## Real-World Example: CalorieSnap App

Let's walk through how Luna Memory transforms a basic calorie tracking app into a sophisticated personalization engine.

### Traditional Implementation vs Luna Memory

**Traditional Approach:**
```sql
-- Standard database schema
CREATE TABLE food_logs (
    user_id TEXT,
    food_item TEXT,
    calories INTEGER,
    timestamp DATETIME,
    image_url TEXT
);

-- Typical queries
SELECT food_item, COUNT(*) as frequency 
FROM food_logs 
WHERE user_id = 'user_123' 
GROUP BY food_item 
ORDER BY frequency DESC;
```

**Problems with Traditional Approach:**
- Fixed categories and rigid structure
- No understanding of relationships between foods
- Can't adapt to new business opportunities
- Limited personalization capabilities

### Luna Memory Enhancement

#### 1. Learn Anything - Learning User Patterns

```json
POST /learn
{
    "description": "Learn user's food preferences, eating patterns, and identify opportunities for personalized recipe recommendations",
    "schema": {
        "attributes": [
            "flavor_preference",
            "eating_pattern",
            "food_relationship"
        ],
        "relationships": [
            "enjoys_with",
            "substitutes_for",
            "avoids_after"
        ]
    }
}

// Example interaction data
{
    "user_id": "user_123",
    "interactions": [
        {
            "timestamp": "2024-03-15T12:30:00Z",
            "food_item": "Grilled Salmon",
            "calories": 400,
            "image_url": "https://...",
            "tags": ["dinner", "protein", "homemade"]
        }
    ]
}

// What Luna learns automatically
{
    "insights": {
        "flavor_preference": {
            "primary": ["umami", "fresh", "citrus"],
            "avoided": ["overly_sweet", "heavily_spiced"]
        },
        "eating_patterns": {
            "peak_protein_times": ["19:00", "12:30"],
            "portion_control": "consistent",
            "meal_complexity": {
                "weekday": "simple",
                "weekend": "elaborate"
            }
        },
        "food_relationships": {
            "frequently_combined": [
                ["salmon", "quinoa"],
                ["avocado", "eggs"]
            ],
            "substitution_patterns": {
                "when_busy": ["meal_prep", "quick_proteins"],
                "when_social": ["restaurant_choices", "home_cooking"]
            }
        }
    }
}
```

#### 2. Ask Anything - Generating User Insights

```json
POST /ask
{
    "user_id": "user_123",
    "question": "Based on this user's patterns, what premium features would they be most likely to subscribe to?",
    "context": {
        "available_features": [
            "meal_planning",
            "recipe_recommendations",
            "nutrition_coaching",
            "shopping_lists"
        ],
        "business_metrics": [
            "engagement_likelihood",
            "predicted_retention",
            "upsell_opportunity"
        ]
    }
}

// Luna's response
{
    "recommendations": [
        {
            "feature": "personalized_meal_planning",
            "confidence": 0.89,
            "reasoning": {
                "primary": "User shows consistent patterns of advance meal preparation and interest in balanced nutrition",
                "supporting_evidence": {
                    "meal_prep_frequency": "72% of weekends",
                    "recipe_interaction_rate": "high",
                    "nutrition_balance_awareness": "demonstrated"
                }
            },
            "predicted_outcomes": {
                "retention_boost": "+25%",
                "health_goal_achievement": "+40%"
            }
        }
    ],
    "graph_insights": {
        "key_nodes_referenced": 12,
        "relationship_patterns": 4,
        "confidence_factors": ["consistent_behavior", "clear_preferences"]
    }
}
```

## Business Value

### 1. Dynamic Learning
- **Traditional:** Fixed categories, manual feature engineering
- **Luna Memory:** Automatically learns new patterns and adapts to user behavior
  - Discovers unexpected food combinations
  - Identifies timing patterns
  - Understands context (busy days vs. leisure cooking)

### 2. Personalization Depth
- **Traditional:** Rule-based recommendations
- **Luna Memory:** Deep understanding of user preferences
  - Links restaurant preferences to home cooking
  - Understands cooking skill progression
  - Adapts to lifestyle changes

### 3. Business Intelligence
- **Traditional:** Predefined reports and metrics
- **Luna Memory:** Dynamic insight generation
  - Identifies upsell opportunities
  - Predicts churn risk
  - Discovers new market segments
