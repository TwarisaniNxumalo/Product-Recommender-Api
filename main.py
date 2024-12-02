from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


data = pd.DataFrame({
    'HairType': ['Dry', 'Frizzy', 'Fine', 'Curly', 'Oily', 'Coarse'],
    'MainConcern': ['Dryness', 'Frizz', 'Volume', 'Curl Definition', 'Scalp Balance', 'Damage'],
    'Texture': ['Medium', 'Coarse', 'Fine', 'Medium', 'Fine', 'Thick'],
    'Goal': ['Hydration', 'Frizz Control', 'Volume', 'Curl Definition', 'Scalp Health', 'Repair'],
    'IngredientPreference': ['Sulfate-Free', 'Paraben-Free', 'Silicone-Free', 'Natural', 'Fragrance-Free',
                             'Cruelty-Free'],
    'WashFrequency': ['Daily', 'Every other day', 'Once a week', 'As needed', 'Every other day', 'Daily'],
    'Product': [
        'Coconut Oil Hydrate Shampoo', 'Hemp Oil Frizz Shampoo', 'Grapefruit Volume Shampoo',
        'Apricot Curl Shampoo', 'Eucalyptus Scalp Balance Shampoo', 'Argan Oil Repair Shampoo'
    ]
})

label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop('Product', axis=1)
y = data['Product']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

questions = [
    {"field": "hair_type", "question": "What is your hair type (e.g., Dry, Frizzy, Fine, etc.)?"},
    {"field": "main_concern", "question": "What is your main hair concern (e.g., Dryness, Frizz, Volume, etc.)?"},
    {"field": "texture",
     "question": "How would you describe the texture of your hair (e.g., Coarse, Medium, Fine, etc.)?"},
    {"field": "goal", "question": "What is your primary hair care goal (e.g., Hydration, Volume, Repair, etc.)?"},
    {"field": "ingredient_pref",
     "question": "Do you have any ingredient preferences (e.g., Sulfate-Free, Paraben-Free, etc.)?"},
    {"field": "wash_frequency", "question": "How often do you wash your hair (e.g., Daily, Once a week, etc.)?"}
]

sessions = {}

# Request model for validation
class UserInput(BaseModel):
    hair_type: str
    main_concern: str
    texture: str
    goal: str
    ingredient_pref: str
    wash_frequency: str

@app.post("/recommend/")
async def recommend_product(user_input: UserInput):
    # Create input DataFrame
    input_data = pd.DataFrame([[user_input.hair_type.strip(), user_input.main_concern.strip(),
                                user_input.texture.strip(), user_input.goal.strip(),
                                user_input.ingredient_pref.strip(), user_input.wash_frequency.strip()]],
                              columns=['HairType', 'MainConcern', 'Texture', 'Goal', 'IngredientPreference', 'WashFrequency'])
    # Validate inputs
    for column in input_data.columns:
        if input_data[column][0] not in label_encoders[column].classes_:
            raise HTTPException(status_code=400, detail=f"Invalid input for {column}: {input_data[column][0]}. "
                                                        f"Expected one of {list(label_encoders[column].classes_)}")

    # Encode inputs
    for column in input_data.columns:
        input_data[column] = label_encoders[column].transform(input_data[column])

    # Predict product
    prediction = model.predict(input_data)[0]
    recommended_product = label_encoders['Product'].inverse_transform([prediction])[0]

    return {"recommended_product": recommended_product}

@app.get("/chatbot/start/")
async def start_chatbot(session_id: str):
    """Initialize a chatbot session."""
    if session_id in sessions:
        del sessions[session_id]
    sessions[session_id] = {"responses": {}, "current_question_index": 0}
    return {"question": questions[0]["question"]}


@app.post("/chatbot/next/")
async def next_question(session_id: str, answer: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID. Start a new session.")

    session = sessions[session_id]
    current_index = session["current_question_index"]
    current_field = questions[current_index]["field"]

    session["responses"][current_field] = answer

    session["current_question_index"] += 1

    if session["current_question_index"] < len(questions):
        next_question = questions[session["current_question_index"]]["question"]
        return {"question": next_question}
    else:

        user_input = session["responses"]

        input_data = pd.DataFrame([[user_input[field] for field in user_input]],
                                  columns=['hair_type', 'main_concern', 'texture', 'goal', 'ingredient_pref',
                                           'wash_frequency'])
        for column in input_data.columns:
            if input_data[column][0] not in label_encoders[column].classes_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input for {column}: {input_data[column][0]}. "
                           f"Expected one of {list(label_encoders[column].classes_)}"
                )
            input_data[column] = label_encoders[column].transform(input_data[column])

        prediction = model.predict(input_data)[0]
        recommended_product = label_encoders['Product'].inverse_transform([prediction])[0]

        del sessions[session_id]

        return {"message": "All questions answered.", "recommended_product": recommended_product}
