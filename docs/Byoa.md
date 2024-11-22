# Luna Memory BYOA Documentation

## Overview

Luna Memory's BYOA (Bring Your Own Algorithm) system allows you to transform static user data into dynamic, business-driving insights. Instead of being limited by predefined schemas and fixed business logic, you can:

- Learn anything about your users through natural language
- Ask complex questions about user behavior and preferences
- Let users define their own personalization algorithms
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

#### 1. Learning User Patterns

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

#### 2. Generating Business Insights

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

#### 3. User-Defined Personalization

```json
POST /byoa/user
{
    "user_id": "user_123",
    "expression": "I want healthy recipes that taste like my favorite restaurant dishes but are easier to make at home",
    "context": {
        "time_available": 30,
        "cooking_skill": "intermediate",
        "kitchen_equipment": ["basic", "instant_pot"]
    }
}

// Luna's personalized response
{
    "recommendations": [
        {
            "recipe": "15-minute Salmon Bowl",
            "match_reasoning": {
                "flavor_profile": "Matches user's preference for Nobu's Black Cod",
                "health_adaptations": [
                    "40% less sodium",
                    "Added vegetables",
                    "Portion-optimized"
                ],
                "convenience_factors": [
                    "Single pan",
                    "Prep time: 15 mins",
                    "Basic ingredients"
                ]
            },
            "user_context": {
                "previous_success": "Has successfully made similar rice bowls",
                "ingredient_familiarity": "85%",
                "predicted_satisfaction": "92%"
            }
        }
    ]
}
```

## Business Value Demonstration

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

### 4. User Empowerment
- **Traditional:** Fixed feature set
- **Luna Memory:** Users define their own algorithms
  - Natural language interface
  - Contextual understanding
  - Adaptive recommendations

## Integration Example

```javascript
// Frontend implementation
async function handleFoodLog(imageData, userId) {
    // 1. Regular calorie logging
    const basicLog = await logFoodItem(imageData);
    
    // 2. Luna Memory enhancement
    const lunaInsights = await luna.learn({
        user_id: userId,
        interaction_type: "food_log",
        data: {
            ...basicLog,
            context: {
                time_of_day: new Date(),
                user_mood: await getUserMoodData(),
                location_type: "home/restaurant"
            }
        }
    });
    
    // 3. Generate personalized recommendations
    const recommendations = await luna.ask({
        user_id: userId,
        question: "What should we recommend to this user right now?",
        context: {
            time_of_day: new Date(),
            recent_interactions: lunaInsights
        }
    });
    
    return {
        basicLog,
        personalizedRecommendations: recommendations
    };
}
```

## Getting Started

1. Get your API key from Luna Memory dashboard
2. Define your initial learning objectives
3. Start sending interaction data
4. Query for insights and recommendations

Need help? Contact us at developer-support@lunamemory.ai