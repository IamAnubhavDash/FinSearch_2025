This project was part of Finsearch iitb which was a 3 months project that deals with Reinforcement learning , deep learning and Finance.
Team Members:
1)Anubhav Dash
2)Maitreya
3)Shivansh Chaudhary
4)Tushar Yadav
Working and Thought Process:
Introduction & Project Goal

"Hello everyone. Today, we're excited to present our project on optimizing stock trading strategies using deep reinforcement learning."

"The financial markets are incredibly complex, and predicting stock movements is a major challenge. Our goal was to see if we could use advanced AI to make smarter trading decisions."

"We developed a Deep Reinforcement Learning model and compared its performance against two other popular methods: LSTM and ARIMA."

"Our primary objective was to maximize returns while managing risk. We used 10 years of Nifty 100 data for our analysis."

Data Preprocessing & Feature Engineering 

"A crucial part of any machine learning project is the data. We started with 10 years of historical data for the Nifty 100 index."

"The raw data needed to be prepared for our models. This involved a few key steps which we handled in our preprocessing.py script."

"First, we calculated important technical indicators like the RSI (Relative Strength Index) to measure market momentum, and 10-day and 50-day Simple Moving Averages to identify trends."

"Then, we scaled all of our data to a common range. This helps our models learn more effectively."

"This preprocessed data formed the foundation for training all three of our models."

Model Deep Dive: ARIMA and LSTM

"Let's now look at our benchmark models. First, the ARIMA model, which is a classical statistical method for time-series forecasting. We implemented this in our train_arima.py script."

"The ARIMA model analyzes past closing prices to forecast future prices. Our trading strategy was simple: if the model predicted a price increase, we'd buy. If it predicted a decrease, we'd sell."

"Next, we have the LSTM model, a more advanced neural network. This was implemented in train_lstm.py."

"LSTMs are great at recognizing patterns in sequential data, like stock prices. We trained our LSTM to predict the next day's closing price based on the previous 50 days of data. The trading logic was similar to our ARIMA model."

"Both of these models provide a solid baseline to compare our main model against."

The Star of the Show: Deep Reinforcement Learning (DQN) 

"Now for the core of our project: the Deep Reinforcement Learning model, specifically a Deep Q-Network or DQN. You can see the implementation in train_dqn.py and env.py."

"Unlike the other models that just predict prices, our DQN agent learns to make decisions. We created a virtual trading environment where the agent could practice buying, holding, and selling stocks."

"For every trade, the agent received a reward or a penalty based on whether it made a profit or a loss. Over 100,000 timesteps, the agent learned a sophisticated trading strategy to maximize its total profit."

"This approach is powerful because the agent learns from its interactions with the market, adapting its strategy as it goes."

Results and Conclusion - Member 1

"So, which model performed the best? Let's look at the results from our evaluation."

"As you can see from this chart, the DQN RL model was the clear winner. It achieved the highest cumulative returns, meaning it was the most profitable."

"It also had the best Sharpe Ratio, which tells us that it generated higher returns for the amount of risk it took. And importantly, it had a lower Maximum Drawdown, indicating it was less risky than the LSTM and ARIMA models."

"In conclusion, our findings strongly suggest that Deep Reinforcement Learning is a highly effective approach for developing sophisticated and profitable automated trading strategies. It outperformed the more traditional LSTM and ARIMA models in both returns and risk management."


