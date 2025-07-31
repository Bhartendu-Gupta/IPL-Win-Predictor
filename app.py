import streamlit as st
import pickle
import pandas as pd

# Load the trained model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Define team and city options
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Streamlit UI setup
st.title("🏏 IPL Win Predictor")

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

# Match location
selected_city = st.selectbox('Select Host City', sorted(cities))

# Match stats input
target = st.number_input('Target Score', min_value=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

# Prediction logic
if st.button('Predict Probability'):
    
    # Basic validation
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same.")
    elif overs == 0:
        st.error("Overs cannot be zero to calculate CRR.")
    else:
        runs_left = max(target - score, 0)
        balls_left = max(120 - int(overs * 6), 0)
        remaining_wickets = 10 - wickets

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [remaining_wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Display match situation
        st.subheader("📊 Match Situation")
        st.dataframe(input_df, use_container_width=True)

        # Handle edge conditions first
        if wickets == 10:
            if score == target:
                st.success("🤝 Match Drawn! All wickets lost, scores are level.")
            elif score > target:
                st.success(f"🎉 {batting_team} chased the target just before losing all wickets!")
            else:
                st.error("All 10 wickets are lost. Innings is over.")
                st.info(f"{bowling_team} has won the match.")
        elif score == target:
            st.balloons()
            st.success(f"🎉 {batting_team} has chased the target! They win!")
        elif balls_left == 0:
            if score > target:
                st.balloons()
                st.success(f"🎉 {batting_team} wins with a thrilling finish!")
            elif score == target:
                st.success("🤝 Match Drawn! Scores are level at the end of 20 overs.")
            else:
                st.error("No balls left. Innings is over.")
                st.info(f"{bowling_team} has won the match.")
        else:
            # Run prediction
            result = pipe.predict_proba(input_df)
            loss_prob = result[0][0]
            win_prob = result[0][1]

            st.subheader("🏆 Win Probability")
            st.success(f"{batting_team} 🔹 {round(win_prob * 100, 2)}%")
            st.info(f"{bowling_team} 🔸 {round(loss_prob * 100, 2)}%")


# Features Covered

# All wickets fallen → match over
# Balls completed → match over
# Negative runs → handled as 0
# Same score with all wickets → match draw
# Exact chase or beyond → success message
# Normal win probability using ML
# Trophy and match icons 🏆📊🏏
# Validations: zero overs, same team check

# ✅ Features Handled

# Same team selected ❌
# Zero overs ❌
# Wickets = 10 ✅
# Overs = 20 ✅
# Score = Target ✅
# Score > Target ✅
# Score < Target ✅
# Final ball win ✅
# Match draw ✅
# Model prediction when valid ✅