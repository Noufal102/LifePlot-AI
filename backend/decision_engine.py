
from datetime import datetime, timedelta
from collections import Counter


# Existing Transformer prediction suggestion
def make_suggestion(activity, confidence, energy, priority):

    if energy <= 2:
        return (
            "Your energy is low. Consider a short break, "
            "water, or a lighter task."
        )

    if activity == "Sleep":
        return (
            "Your pattern suggests rest may be due. "
            "Consider winding down and reducing screen time."
        )

    if activity == "Exercise":
        return (
            "If you feel well, plan a short movement session "
            "and keep it realistic."
        )

    if activity in ("Work", "Study") and priority >= 4:
        return (
            f"Your next likely activity is {activity}. "
            "Try a focused 25-minute session."
        )

    if activity == "Eating":
        return (
            "A meal may fit your routine. Check your hunger "
            "and schedule before deciding."
        )

    return (
        f"Your next likely activity is {activity}. "
        "Review your priorities before switching tasks."
    )


# Daily activity insights
def generate_daily_insights(history):

    now = datetime.now()
    last_24_hours = now - timedelta(hours=24)

    recent_activities = []

    for item in history:
        timestamp = item.timestamp

        if timestamp is None:
            continue

        # Handle timezone-aware and naive timestamps
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        if last_24_hours <= timestamp <= now:
            recent_activities.append(item)

    if not recent_activities:
        return {
            "total_activities": 0,
            "total_minutes": 0,
            "average_energy": 0,
            "high_priority_count": 0,
            "most_common_activity": "Not available",
            "insights": [
                "Add activities to generate your daily insights."
            ],
            "recommendations": [
                "Start by logging your daily activities."
            ]
        }

    total_activities = len(recent_activities)

    total_minutes = sum(
        item.duration for item in recent_activities
    )

    average_energy = round(
        sum(item.energy_level for item in recent_activities)
        / total_activities,
        2
    )

    high_priority_count = sum(
        1 for item in recent_activities
        if item.priority >= 4
    )

    activity_counts = Counter(
        item.activity for item in recent_activities
    )

    most_common_activity = activity_counts.most_common(1)[0][0]

    insights = []
    recommendations = []

    # Activity duration analysis
    if total_minutes >= 600:
        insights.append(
            "You have logged 10 or more hours of activity "
            "during the last 24 hours."
        )
        recommendations.append(
            "Review your schedule and make room for sufficient rest."
        )
    else:
        insights.append(
            f"You logged {total_minutes} minutes of activity "
            "during the last 24 hours."
        )

    # Energy analysis
    if average_energy <= 2:
        insights.append(
            "Your average recorded energy level is low."
        )
        recommendations.append(
            "Consider lighter tasks and short breaks."
        )
    elif average_energy >= 4:
        insights.append(
            "Your average recorded energy level is high."
        )
        recommendations.append(
            "Consider using your available energy for important tasks."
        )
    else:
        insights.append(
            "Your recorded energy level is moderate."
        )

    # Priority analysis
    if high_priority_count > 0:
        insights.append(
            f"You logged {high_priority_count} high-priority activities."
        )
        recommendations.append(
            "Review high-priority tasks and decide which needs attention first."
        )
    else:
        insights.append(
            "No high-priority activities were recorded."
        )
        recommendations.append(
            "Consider identifying your most important task for the day."
        )

    # Activity pattern
    insights.append(
        f"Your most frequently logged activity is {most_common_activity}."
    )

    recommendations.append(
        "Continue logging activities to improve your routine analysis."
    )

    return {
        "total_activities": total_activities,
        "total_minutes": total_minutes,
        "average_energy": average_energy,
        "high_priority_count": high_priority_count,
        "most_common_activity": most_common_activity,
        "insights": insights,
        "recommendations": recommendations
    }
