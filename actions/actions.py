from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import SlotSet
import logging

logger = logging.getLogger((__name__))

logging.basicConfig(level=logging.DEBUG)

INTENT_TO_EMOTION = {
    "sad": "sad",
    "depressed": "depressed",
    "stressed": "stressed",
    "anxious": "anxious",
    "overwhelmed": "overwhelmed",
    "lonely": "lonely",
    "hurt": "hurt",
    "hopeless": "hopeless",
    "worried": "worried",
    "angry": "angry",
}

class ActionGetSentiment(Action):
    def name(self) -> Text:
        return "action_get_sentiment"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        latest_intent = tracker.latest_message.get("intent", {}).get("name")
        emotion = INTENT_TO_EMOTION.get(latest_intent)

        logger.debug(f"[ActionGetSentiment] Detected intent: {latest_intent} -> emotion: {emotion}")

        events: List[Dict[Text, Any]] = []
        if emotion:
            events.append(SlotSet("last_emotion", emotion))
            # Set depression flag
            events.append(SlotSet("is_depressed", emotion == "depressed"))
        return events

import random
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet


class ActionSetEmotionalIntent(Action):
    def name(self) -> Text:
        return "action_set_emotional_intent"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> List[Dict[Text, Any]]:
        latest_intent = tracker.latest_message['intent'].get('name')

        intent_to_emotion = {
            "sad": "sad",
            "depressed": "depressed",
            "stressed": "stressed",
            "anxious": "anxious",
            "overwhelmed": "overwhelmed",
            "lonely": "lonely",
            "hurt": "hurt",
            "hopeless": "hopeless",
            "worried": "worried",
            "angry": "angry"
        }

        emotion = intent_to_emotion.get(latest_intent)

        if emotion:
            dispatcher.utter_message(text=f"I hear that you're feeling {emotion}.")
            return [
                SlotSet("last_emotion", emotion),
                SlotSet("is_depressed", emotion == "depressed")
            ]

        return []


class ActionProvideCopingMechanism(Action):
    def name(self) -> Text:
        return "action_provide_coping_mechanism"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        coping_responses = {
            "sad": [
                "I'm sorry you're feeling sad. Sometimes writing your thoughts in a journal or listening to calming music can help.",
                "Feeling sad is tough. A short walk or talking to someone you trust might lift your mood.",
                "Sadness can be overwhelming. Try listening to soothing music or expressing yourself through writing."
            ],
            "depressed": [
                "It sounds like you're going through a really heavy time. Reaching out to a trusted friend or practicing small self-care steps might help.",
                "During depressive moments, gentle routines like a short walk, journaling, or connecting with someone supportive can help.",
                "I know it's hard. Small steps, like breathing exercises or talking to a friend, can make a difference."
            ],
            "stressed": [
                "Stress can feel overwhelming. Try taking short breaks, deep breathing exercises, or a quick walk.",
                "When stressed, pausing for a moment and stretching, meditating, or journaling can help you regain calm.",
                "Feeling stressed? Deep breaths, listening to music, or talking to someone can help ease it."
            ],
            "anxious": [
                "Anxiety is tough. Grounding techniques like focusing on 5 things you can see, hear, and feel can calm your mind.",
                "Try deep breathing or journaling to manage anxious thoughts. Small steps can make a difference.",
                "Anxiety can feel heavy. Practicing mindfulness or talking to someone supportive can help."
            ],
            "overwhelmed": [
                "Feeling overwhelmed happens. Break tasks into smaller steps and give yourself permission to rest.",
                "Overwhelm can be exhausting. Take short breaks, prioritize one task at a time, and breathe.",
                "It’s okay to feel overwhelmed. Doing small, manageable steps and resting in between can help."
            ],
            "lonely": [
                "Loneliness can be hard. Maybe call or message someone you trust, or join an online community of interest.",
                "Feeling lonely? Reaching out to a friend or joining a supportive group online might help.",
                "Loneliness is challenging. Consider connecting with someone you trust or doing a hobby you enjoy."
            ],
            "hurt": [
                "I'm sorry you feel hurt. Sometimes talking it out with someone supportive or writing it down can help release it.",
                "Feeling hurt is normal. Sharing your feelings with a trusted person or journaling can ease it.",
                "Hurt can linger. Gentle self-expression or seeking support can be helpful."
            ],
            "hopeless": [
                "Feeling hopeless is really heavy. Try focusing on one small thing you can control right now, and remember that feelings pass.",
                "Hopelessness can be overwhelming. Taking small positive actions or talking to someone supportive may help.",
                "When you feel hopeless, remember tiny steps and small victories matter. You’re not alone."
            ],
            "worried": [
                "Worrying can spiral fast. Try writing your worries down and challenging whether they’re facts or ‘what ifs’.",
                "Feeling worried? Deep breathing and listing practical next steps can calm your mind.",
                "Worries can take over. Consider talking them out or journaling to gain perspective."
            ],
            "angry": [
                "Anger can feel powerful. Physical release like exercise or even squeezing a stress ball might help you let it out safely.",
                "Feeling angry? Try going for a run, punching a pillow, or talking it out with someone you trust.",
                "Anger can be intense. Deep breathing, movement, or expressing it safely can help."
            ],
            "suicidal": [
                "Please know that you are not alone and there are people who can help. If you're having suicidal thoughts, please reach out to a professional immediately. You can contact Snehi at 91-9582208181 or Aasra at 91-9820466726 (India), or find international hotlines at https://www.iasp.info/resources/Crisis_Centres/. 💙 Would you like me to share some ways to calm your mind right now?",
                "I’m really concerned about your safety. 💙 Remember that help is available. If you’re in immediate danger, please call your local emergency services. Snehi: 91-9582208181, Aasra: 91-9820466726 (India), or international hotlines: https://www.iasp.info/resources/Crisis_Centres/. Want to try a small calming exercise now?",
                "You’re not alone — support is here. 💙 Reach out to Snehi (91-9582208181) or Aasra (91-9820466726) in India, or see international hotlines at https://www.iasp.info/resources/Crisis_Centres/. Shall we do a short calming activity together?"
            ]
        }

        emotional_keywords = {
            "sad": ["sad", "unhappy", "down", "blue"],
            "stressed": ["stressed", "overwhelmed", "pressure", "burned out"],
            "depressed": ["depressed", "hopeless", "downcast"],
            "anxious": ["anxious", "nervous", "worried", "tense"],
            "overwhelmed": ["overwhelmed", "swamped"],
            "lonely": ["lonely", "isolated", "alone"],
            "hurt": ["hurt", "betrayed", "pain"],
            "hopeless": ["hopeless", "desperate"],
            "worried": ["worried", "concerned", "apprehensive"],
            "angry": ["angry", "mad", "furious", "irritated"]
        }

        suicidal_keywords = ["kill myself", "hurt myself", "suicide", "end my life"]


        user_message = tracker.latest_message.get("text", "").lower()
        last_emotion = tracker.get_slot("last_emotion")

        if any(word in user_message for word in suicidal_keywords):
            dispatcher.utter_message(text=random.choice(coping_responses["suicidal"]))
            return [SlotSet("last_emotion", "suicidal")]

        detected_emotion = None
        for emotion, keywords in emotional_keywords.items():
            if any(word in user_message for word in keywords):
                detected_emotion = emotion
                break

        if detected_emotion:
            dispatcher.utter_message(text=random.choice(coping_responses[detected_emotion]))
            return [SlotSet("last_emotion", detected_emotion)]
        elif last_emotion:
            dispatcher.utter_message(text=random.choice(coping_responses[last_emotion]))
            return []
        else:
            dispatcher.utter_message(
                text="Here are some general coping mechanisms: deep breathing, journaling, short walks, or talking to someone you trust."
            )
            return []


class ActionLearnMore(Action):
    def name(self) -> Text:
        return "action_learn_more"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="I can tell you more about mental health facts. What would you like to know?")
        return []

class ActionProvideSleepTips(Action):
    def name(self) -> Text:
        return "action_provide_sleep_tips"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(
            text="Try creating a relaxing bedtime routine. Turn off screens an hour before bed and try reading a book.")
        return []


class ActionLowConfidence(Action):
    def name(self) -> Text:
        return "action_low_confidence"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        messages = [
            "I'm sorry, I didn't quite understand that. Could you please rephrase?",
            "I'm not sure what you mean. Can you say that in a different way?",
            "Hmm, I didn’t get that. Would you mind rephrasing?",
            "Sorry, I’m having trouble following. Could you explain again?",
            "I might have misunderstood you — can you put that another way?"
        ]

        response = random.choice(messages)
        dispatcher.utter_message(text=response)
        return []


class ProvideResourcesForm(FormValidationAction):
    def name(self) -> Text:
        return "provide_resources_form"

    async def required_slots(
            self,
            slots: Dict[Text, Any],
            tracker: Tracker
    ) -> List[Text]:
        return ["is_depressed"]

    async def extract_is_depressed(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        intent_name = tracker.get_intent_of_last_message()
        if intent_name == "affirm":
            return {"is_depressed": True}
        elif intent_name == "deny":
            return {"is_depressed": False}
        else:
            return {"is_depressed": None}


class ActionProvideResources(Action):
    def name(self) -> Text:
        return "action_provide_resources"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        is_depressed = tracker.get_slot("is_depressed")
        if is_depressed:
            dispatcher.utter_message(response="utter_provide_resources")
        else:
            dispatcher.utter_message(text="Okay, I understand. I'm here if you change your mind.")
        return [SlotSet("is_depressed", None)]


class ActionProvideMentalHealthFact(Action):
    def name(self) -> Text:
        return "action_provide_mental_health_fact"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        mental_health_facts = {
            "mental health": "Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel, and act.",
            "mental illness": "Having a mental illness means having a condition that affects your thinking, feeling, or mood and can affect your ability to relate to others and function each day.",
            "depression": "Depression is a common and serious medical illness that negatively affects how you feel, the way you think, and how you act.",
            "therapist": "A therapist is a trained professional who can help you work through emotional and mental challenges.",
            "therapy": "Therapy is a form of treatment that helps people with a broad range of mental illnesses and emotional difficulties. It is for anyone who wants to improve their mental health.",
            "sadness": "Sadness is a temporary emotion that we all feel from time to time, while depression is a persistent mental health condition that affects your mood and ability to function.",
            "anxiety": "Stress is a normal reaction to a difficult situation, while anxiety is a mental health condition characterized by excessive worry, even in the absence of a stressor.",
            "mental disorder": "A mental disorder is a condition that affects your thinking, feeling or mood and can affect your ability to relate to others and function each day.",
            "treatment options": "Treatment options include therapy, medication, and support groups. A professional can help you find the best option for you.",
            "medication": "Before starting a new medication, it’s important to consult a doctor about side effects, interactions, and proper use.",
            "social connections": "To maintain social connections, you can reach out to friends and family, join a club or group, or volunteer. If you feel lonely, remember you're not alone and many people feel this way.",
            "unwell": "It's natural to feel unwell sometimes. However, if symptoms are persistent or interfere with your daily life, it may be a good idea to seek professional help.",
            "support group": "You can find a support group through local community centers, hospitals, or online platforms.",
            "mental health professional": "Finding the right professional often involves research. You can ask about their experience, specialties, and approach to therapy.",
            "help": "You can also get help from support groups, community mental health centers, or national organizations like NAMI."
        }

        synonym_mapping = {
            "depressed": "depression",
            "mentally ill": "mental illness",
            "ill": "mental illness",
            "stress": "anxiety",
            "disorder": "mental disorder",
            "treatments": "treatment options",
            "professionals": "mental health professional"
        }

        topic = tracker.get_slot("mental_health_topic_slot")

        if topic:
            topic = topic.lower()
        else:
            topic = ""

        mapped_topic = synonym_mapping.get(topic, topic)

        fact_text = mental_health_facts.get(
            mapped_topic,
            f"I'm not sure about '{topic}', but I can share general mental health resources if you'd like."
        )

        dispatcher.utter_message(text=fact_text)

        return [SlotSet("mental_health_topic_slot", None)]

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionEndConversation(Action):
    def name(self) -> Text:
        return "action_end_conversation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Take care of yourself. 💙")
        return []