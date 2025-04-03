Week 1: Basic Python Programming - "Banking Basics Bootcamp"Goal: Learn Python fundamentals through simple banking-related tasks.# Exercise 1: Calculate Simple Interest - "Grow Your Savings!"
def calculate_simple_interest(principal, rate, time):
    interest = (principal * rate * time) / 100
    return interest

# Example usage
principal = 1000  # dollars
rate = 5  # percent
time = 2  # years
interest = calculate_simple_interest(principal, rate, time)
print(f"Interest earned: ${interest:.2f}")

# Interactive part
user_principal = float(input("Enter your principal amount: $"))
user_rate = float(input("Enter the annual interest rate (%): "))
user_time = int(input("Enter the time period in years: "))
user_interest = calculate_simple_interest(user_principal, user_rate, user_time)
print(f"Your interest earned: ${user_interest:.2f}")

# Exercise 2: Total and Average Expenses - "Track Your Spending!"
expenses = [100, 200, 150, 300, 250]  # dollars
total_expenses = sum(expenses)
average_expense = total_expenses / len(expenses)
print(f"Total expenses: ${total_expenses}")
print(f"Average expense: ${average_expense:.2f}")

# Interactive part
user_expenses = []
for i in range(5):
    expense = float(input(f"Enter expense {i+1}: $"))
    user_expenses.append(expense)
user_total = sum(user_expenses)
user_average = user_total / len(user_expenses)
print(f"Your total expenses: ${user_total}")
print(f"Your average expense: ${user_average:.2f}")

# Exercise 3: Currency Conversion - "Travel Ready!"
def convert_currency(amount, rate):
    return amount * rate

# Example usage
usd_amount = 100
exchange_rate = 0.85  # USD to EUR
eur_amount = convert_currency(usd_amount, exchange_rate)
print(f"${usd_amount} USD is €{eur_amount:.2f} EUR")

# Interactive part
user_amount = float(input("Enter amount in USD: $"))
user_rate = float(input("Enter exchange rate (USD to EUR): "))
user_eur = convert_currency(user_amount, user_rate)
print(f"${user_amount} USD is €{user_eur:.2f} EUR")Notes: This week introduces variables, functions, loops, and user input with practical banking examples like interest calculation and expense tracking.Week 2: Loops and Control Structures - "Transaction Detective"Goal: Master loops and conditionals by analyzing transaction-like data.# Sample transaction data
transactions = [100, -50, 200, -100, 50, -20]

# Exercise 1: Count Deposits and Withdrawals - "Spot the Cash Flow!"
deposits = 0
withdrawals = 0
for t in transactions:
    if t > 0:
        deposits += 1
    elif t < 0:
        withdrawals += 1
print(f"Deposits: {deposits}, Withdrawals: {withdrawals}")

# Exercise 2: Find Max and Min Transactions - "Biggest Spender Award!"
max_trans = transactions[0]
min_trans = transactions[0]
for t in transactions:
    if t > max_trans:
        max_trans = t
    if t < min_trans:
        min_trans = t
print(f"Largest deposit: ${max_trans}")
print(f"Largest withdrawal: ${min_trans}")

# Exercise 3: Age Categories - "Customer Segmentation!"
ages = [25, 30, 22, 40, 35, 28]
categories = []
for age in ages:
    if age <= 25:
        categories.append('Young Savers (18-25)')
    elif age <= 35:
        categories.append('Prime Earners (26-35)')
    else:
        categories.append('Wise Investors (36+)')
print("Customer Segments:", categories)Notes: Focuses on loops and conditionals to process lists, teaching iteration and decision-making with banking-related data.Week 3: Functions and Modules - "Banking Toolkit"Goal: Create reusable functions and introduce NumPy for numerical tasks.import numpy as np

# Exercise 1: Function to Compute Total Balance - "Balance Calculator!"
def compute_total_balance(transactions):
    return sum(transactions)

# Example usage
sample_transactions = [100, -50, 200, -100, 50, -20]
total = compute_total_balance(sample_transactions)
print(f"Total balance: ${total}")

# Exercise 2: NumPy for Stats - "Transaction Insights!"
data = np.array([100, 200, 150, 300, 250])
mean = np.mean(data)
std_dev = np.std(data)
print(f"Average transaction: ${mean:.2f}")
print(f"Transaction variability (std dev): ${std_dev:.2f}")

# Exercise 3: Banking Module - "Loan Planner!"
# Save this function in a separate file named 'banking.py'
"""
# banking.py
def calculate_loan_payment(principal, rate, time):
    return principal * (1 + (rate / 100) * time)
"""

# Main script (assuming banking.py exists)
try:
    from banking import calculate_loan_payment
    payment = calculate_loan_payment(10000, 5, 2)
    print(f"Total repayment for a $10,000 loan over 2 years at 5%: ${payment:.2f}")
except ImportError:
    # Fallback if banking.py is not created
    def calculate_loan_payment(principal, rate, time):
        return principal * (1 + (rate / 100) * time)
    payment = calculate_loan_payment(10000, 5, 2)
    print(f"Total repayment for a $10,000 loan over 2 years at 5%: ${payment:.2f}")Notes: Introduces function definitions, module imports, and NumPy for efficient calculations, with a modular loan planner as a practical example.Week 4: Pandas Operations - "Data Dive with Pandas"Goal: Load and manipulate banking data using pandas.import pandas as pd

# Simulate sample data (in real use, load from CSV files)
customers_data = {'CustomerID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 35, 45]}
accounts_data = {'AccountNumber': [1, 2, 3], 'CustomerID': [1, 2, 3], 'AccountType': ['Savings', 'Checking', 'Savings'], 'CurrentBalance': [1000, 500, 2000]}
customers = pd.DataFrame(customers_data)
accounts = pd.DataFrame(accounts_data)

# Exercise 1: Explore Customers - "Meet Your Clients!"
print("Customer DataFrame Info:")
print(customers.info())
print("\nCustomer DataFrame Description:")
print(customers.describe())

# Exercise 2: Merge Data - "Connect the Dots!"
customer_accounts = pd.merge(customers, accounts, on='CustomerID')
print("Merged Customer-Account DataFrame:")
print(customer_accounts[['Name', 'AccountNumber', 'AccountType', 'CurrentBalance']].head())

# Exercise 3: Analyze by Account Type - "Balance Breakdown!"
avg_balance_by_type = accounts.groupby('AccountType')['CurrentBalance'].mean()
print("Average Balance by Account Type:")
print(avg_balance_by_type)Notes: Teaches pandas basics—loading data, merging DataFrames, and grouping—using simulated banking data. In practice, replace the simulated data with pd.read_csv().Week 5: Matplotlib Visualization - "Picture Your Data"Goal: Visualize banking data with Matplotlib.import matplotlib.pyplot as plt
import pandas as pd

# Reuse sample data from Week 4
customers_data = {'CustomerID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 35, 45]}
accounts_data = {'AccountNumber': [1, 2, 3], 'CustomerID': [1, 2, 3], 'AccountType': ['Savings', 'Checking', 'Savings'], 'CurrentBalance': [1000, 500, 2000]}
transactions_data = {'AccountNumber': [1, 1, 1], 'TransactionDate': ['2023-01-01', '2023-02-01', '2023-03-01'], 'BalanceAfterTransaction': [1000, 900, 950]}
customers = pd.DataFrame(customers_data)
accounts = pd.DataFrame(accounts_data)
transactions = pd.DataFrame(transactions_data)

# Exercise 1: Customer Age Histogram - "Age Demographics!"
plt.hist(customers['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Ages')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.show()

# Exercise 2: Account Type Bar Chart - "Account Popularity!"
account_types = accounts['AccountType'].value_counts()
plt.bar(account_types.index, account_types.values, color='lightgreen')
plt.title('Number of Accounts by Type')
plt.xlabel('Account Type')
plt.ylabel('Count')
plt.show()

# Exercise 3: Transaction Time Series - "Track Account 1!"
acc1_transactions = transactions[transactions['AccountNumber'] == 1].sort_values('TransactionDate')
plt.plot(acc1_transactions['TransactionDate'], acc1_transactions['BalanceAfterTransaction'], color='purple')
plt.title('Balance Over Time for Account 1')
plt.xlabel('Date')
plt.ylabel('Balance ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()Notes: Introduces Matplotlib for creating histograms, bar charts, and time series plots, enhancing data interpretation skills.Week 6: Exploratory Data Analysis (EDA) - "Banking Insights Unveiled"Goal: Perform full EDA on banking data.import seaborn as sns
import pandas as pd
import numpy as np

# Reuse and expand sample data
customers_data = {'CustomerID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 35, 45]}
accounts_data = {'AccountNumber': [1, 2, 3], 'CustomerID': [1, 2, 3], 'CurrentBalance': [1000, 500, 2000]}
transactions_data = {'TransactionAmount': [100, -50, 200], 'AccountNumber': [1, 2, 3]}
loans_data = {'LoanType': ['Personal', 'Mortgage', 'Personal'], 'LoanAmount': [5000, 100000, 3000]}
customers = pd.DataFrame(customers_data)
accounts = pd.DataFrame(accounts_data)
transactions = pd.DataFrame(transactions_data)
loans = pd.DataFrame(loans_data)

# Exercise 1: Transaction Summary - "Know Your Transactions!"
print("Transaction Amount Statistics:")
print(transactions['TransactionAmount'].describe())

# Exercise 2: Handle Missing Data - "Clean It Up!"
transactions_with_na = transactions.copy()
transactions_with_na.loc[0:1, 'TransactionAmount'] = np.nan
print("Rows with missing values:", transactions_with_na['TransactionAmount'].isna().sum())
transactions_clean = transactions_with_na.dropna()
print("Data cleaned! Rows remaining:", len(transactions_clean))

# Exercise 3: Correlation Analysis - "Age vs. Wealth!"
total_balance = accounts.groupby('CustomerID')['CurrentBalance'].sum().reset_index()
customer_data = pd.merge(customers, total_balance, on='CustomerID')
correlation = customer_data['Age'].corr(customer_data['CurrentBalance'])
print(f"Correlation between Age and Total Balance: {correlation:.3f}")

# Exercise 4: Loan Amount Visualization - "Loan Insights!"
sns.boxplot(x='LoanType', y='LoanAmount', data=loans, palette='pastel')
plt.title('Loan Amounts by Loan Type')
plt.xlabel('Loan Type')
plt.ylabel('Loan Amount ($)')
plt.show()Notes: Combines pandas, NumPy, and Seaborn for a comprehensive EDA, including statistics, data cleaning, correlation, and visualization.Week 7: Basic Statistics - "Banking by the Numbers"Goal: Learn and apply basic statistical concepts to banking data.import pandas as pd
import numpy as np
from scipy import stats

# Reuse sample data
customers_data = {'CustomerID': [1, 2, 3, 4, 5], 'Age': [25, 35, 45, 30, 40]}
transactions_data = {'TransactionAmount': [100, -50, 200, 150, 75]}
customers = pd.DataFrame(customers_data)
transactions = pd.DataFrame(transactions_data)

# Exercise 1: Mean, Median, Mode - "Central Tendencies!"
transaction_amounts = transactions['TransactionAmount']
mean_amount = np.mean(transaction_amounts)
median_amount = np.median(transaction_amounts)
mode_amount = stats.mode(transaction_amounts)[0][0]
print(f"Mean transaction amount: ${mean_amount:.2f}")
print(f"Median transaction amount: ${median_amount:.2f}")
print(f"Mode transaction amount: ${mode_amount:.2f}")

# Exercise 2: Standard Deviation and Variance - "Spread of Transactions!"
std_dev = np.std(transaction_amounts)
variance = np.var(transaction_amounts)
print(f"Standard deviation: ${std_dev:.2f}")
print(f"Variance: ${variance:.2f}")

# Exercise 3: Hypothesis Testing - "Age and Spending!"
customer_ages = customers['Age']
t_stat, p_value = stats.ttest_1samp(customer_ages, 40)
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
if p_value < 0.05:
    print("The mean age is significantly different from 40.")
else:
    print("The mean age is not significantly different from 40.")

# Exercise 4: Probability Distributions - "Transaction Normality!"
stat, p = stats.normaltest(transaction_amounts)
print(f"Normality test statistic: {stat:.3f}, P-value: {p:.3f}")
if p < 0.05:
    print("Transaction amounts are not normally distributed.")
else:
    print("Transaction amounts are normally distributed.")
  Notes: Introduces statistics—central tendencies, spread, hypothesis testing, and distribution checks—using banking data for practical application.SummaryWeeks 1-3: Build Python basics, loops, functions, and NumPy skills with simple banking tasks.Weeks 4-6: Advance to pandas, Matplotlib, and EDA for data manipulation and visualization.Week 7: Add basic statistics to analyze banking data statistically.Each week's code is interactive, encouraging learners to input data and explore. For Weeks 4-7, the sample data is simulated; in a real scenario, replace with pd.read_csv() to load actual CSV files. This structure ensures a progressive, engaging learning experience rooted in banking contexts

Week 1: Basic Python Programming - "Banking Basics Bootcamp"Goal: Learn Python fundamentals through simple banking-related tasks.print("Welcome to Week 1: Banking Basics Bootcamp!")

# Exercise 1: Calculate Simple Interest - "Grow Your Savings!"
principal = 1000  # Starting amount in dollars
rate = 5         # Annual interest rate in percent
time = 2         # Time in years
interest = (principal * rate * time) / 100
print(f"Congrats! Your savings of ${principal} earned ${interest} in interest over {time} years!")

# Exercise 2: Total and Average Expenses - "Track Your Spending!"
expenses = [100, 200, 150, 300, 250]  # Daily expenses in dollars
total = sum(expenses)
average = total / len(expenses)
print(f"Your total spending this week was ${total}!")
print(f"On average, you spent ${average:.2f} per day. Smart budgeting!")

# Exercise 3: Currency Conversion - "Travel Ready!"
usd_amount = 100
exchange_rate = 0.85  # USD to EUR
eur_amount = usd_amount * exchange_rate
print(f"Heading to Europe? Your ${usd_amount} USD converts to €{eur_amount:.2f} EUR!")Motivation: Start with fun, relatable tasks like growing savings or planning a trip to build confidence in Python basics.Week 2: Loops and Control Structures - "Transaction Detective"Goal: Master loops and conditionals by analyzing transaction-like data.print("Welcome to Week 2: Transaction Detective!")

# Sample transaction data
transactions = [100, -50, 200, -100, 50, -20]

# Exercise 1: Count Deposits and Withdrawals - "Spot the Cash Flow!"
deposits = 0
withdrawals = 0
for t in transactions:
    if t > 0:
        deposits += 1
    elif t < 0:
        withdrawals += 1
print(f"Detective Report: {deposits} deposits and {withdrawals} withdrawals found!")

# Exercise 2: Find Max and Min Transactions - "Biggest Spender Award!"
max_trans = transactions[0]
min_trans = transactions[0]
for t in transactions:
    if t > max_trans:
        max_trans = t
    if t < min_trans:
        min_trans = t
print(f"Biggest deposit: ${max_trans}")
print(f"Biggest withdrawal: ${min_trans}")

# Exercise 3: Age Categories - "Customer Segmentation!"
ages = [25, 30, 22, 40, 35, 28]
categories = []
for age in ages:
    if age <= 25:
        categories.append('Young Savers (18-25)')
    elif age <= 35:
        categories.append('Prime Earners (26-35)')
    else:
        categories.append('Wise Investors (36+)')
print("Customer Segments:", categories)Fun Factor: Play the role of a "Transaction Detective" to uncover patterns, making loops exciting and interactive.Week 3: Functions and Modules - "Banking Toolkit"Goal: Create reusable functions and explore NumPy with banking scenarios.print("Welcome to Week 3: Banking Toolkit!")

# Exercise 1: Function to Compute Total Balance - "Balance Calculator!"
def compute_total_balance(transactions):
    return sum(transactions)

sample_transactions = [100, -50, 200, -100, 50, -20]
total = compute_total_balance(sample_transactions)
print(f"Your account balance is ${total}. Keep it growing!")

# Exercise 2: NumPy for Stats - "Transaction Insights!"
import numpy as np
data = np.array([100, 200, 150, 300, 250])
mean = np.mean(data)
std_dev = np.std(data)
print(f"Average transaction: ${mean:.2f}")
print(f"Transaction variability (std dev): ${std_dev:.2f}")

# Exercise 3: Banking Module - "Loan Planner!"
# Save this in a file named 'banking.py'
"""
def calculate_loan_payment(principal, rate, time):
    return principal * (1 + (rate / 100) * time)
"""

# Main script
from banking import calculate_loan_payment
payment = calculate_loan_payment(10000, 5, 2)
print(f"Total repayment for a $10,000 loan over 2 years at 5%: ${payment:.2f}")Engagement: Build a "Banking Toolkit" with functions, introducing modularity and NumPy for quick insights.Week 4: Pandas Operations - "Data Dive with Pandas"Goal: Load and manipulate banking data using pandas.import pandas as pd

print("Welcome to Week 4: Data Dive with Pandas!")

# Load the generated data
customers = pd.read_csv('customers.csv')
accounts = pd.read_csv('accounts.csv')
transactions = pd.read_csv('transactions.csv')
loans = pd.read_csv('loans.csv')

# Exercise 1: Explore Customers - "Meet Your Clients!"
print("Here's a peek at your banking customers:")
print(customers.head())

# Exercise 2: Merge Data - "Connect the Dots!"
customer_accounts = pd.merge(customers, accounts, on='CustomerID')
print("Customer-Account Snapshot:")
print(customer_accounts[['Name', 'AccountNumber', 'AccountType', 'CurrentBalance']].head())

# Exercise 3: Analyze by Account Type - "Balance Breakdown!"
avg_balance_by_type = accounts.groupby('AccountType')['CurrentBalance'].mean()
print("Average Balance by Account Type:")
print(avg_balance_by_type)Motivation: Dive into real banking data, connecting customers to accounts like a data scientist in action!Week 5: Matplotlib Visualization - "Picture Your Data"Goal: Visualize banking data with Matplotlib for impactful insights.import matplotlib.pyplot as plt

print("Welcome to Week 5: Picture Your Data!")

# Exercise 1: Customer Age Histogram - "Age Demographics!"
plt.hist(customers['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Ages')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.show()

# Exercise 2: Account Type Bar Chart - "Account Popularity!"
account_types = accounts['AccountType'].value_counts()
plt.bar(account_types.index, account_types.values, color='lightgreen')
plt.title('Number of Accounts by Type')
plt.xlabel('Account Type')
plt.ylabel('Count')
plt.show()

# Exercise 3: Transaction Time Series - "Track Account 1!"
acc1_transactions = transactions[transactions['AccountNumber'] == 1].sort_values('TransactionDate')
plt.plot(acc1_transactions['TransactionDate'], acc1_transactions['BalanceAfterTransaction'], color='purple')
plt.title('Balance Over Time for Account 1')
plt.xlabel('Date')
plt.ylabel('Balance ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()Fun Factor: Turn numbers into colorful charts, making data come alive with visual storytelling.Week 6: Exploratory Data Analysis (EDA) - "Banking Insights Unveiled"Goal: Perform full EDA on banking data, combining all skills learned.import seaborn as sns

print("Welcome to Week 6: Banking Insights Unveiled!")

# Exercise 1: Transaction Summary - "Know Your Transactions!"
print("Transaction Amount Stats:")
print(transactions['TransactionAmount'].describe())

# Exercise 2: Handle Missing Data - "Clean It Up!"
# Simulate missing data
transactions_with_na = transactions.copy()
transactions_with_na.loc[0:5, 'TransactionAmount'] = np.nan
print("Rows with missing values:", transactions_with_na['TransactionAmount'].isna().sum())
transactions_clean = transactions_with_na.dropna()
print("Data cleaned! Rows remaining:", len(transactions_clean))

# Exercise 3: Correlation Analysis - "Age vs. Wealth!"
total_balance = accounts.groupby('CustomerID')['CurrentBalance'].sum().reset_index()
customer_data = pd.merge(customers, total_balance, on='CustomerID')
correlation = customer_data['Age'].corr(customer_data['CurrentBalance'])
print(f"Correlation between Age and Total Balance: {correlation:.3f}")

# Exercise 4: Loan Amount Visualization - "Loan Insights!"
sns.boxplot(x='LoanType', y='LoanAmount', data=loans, palette='pastel')
plt.title('Loan Amounts by Loan Type')
plt.xlabel('Loan Type')
plt.ylabel('Loan Amount ($)')
plt.show()

  # -*- coding: utf-8 -*-
"""
Banking Data Science Training Program - 6 Weeks to EDA Mastery
Created by: Data Adventure Island Team
Special Features:
- Realistic Banking Data Generation
- Interactive Challenges
- Progress Tracking
- Gamified Learning
"""

# ========================
# Week 1: Python Basics & Banking Introduction
# ========================
print("\n=== WEEK 1: Python Basics & Banking Introduction ===")

# Module 1: Banking Story Introduction
print("""\nWelcome to Data Adventure Island!
You've just joined MegaBank International as a junior data analyst.
Your mission: Help solve critical banking problems through data analysis!""")

# Interactive Banking Quiz
score = 0
print("\n** Quick Banking Knowledge Check **")
answers = {
    1: 'loan',
    2: 'savings',
    3: 'credit score'
}

for q in range(1, 4):
    user_answer = input(f"\nQ{q}: What do customers typically need for big purchases? "
                       f"What account earns interest? "
                       f"What determines loan eligibility? ").lower()
    if user_answer in answers[q]:
        score += 1
        print("💵 Correct! Financial knowledge detected!")
    else:
        print("📉 Hmm... Let's learn more!")

print(f"\nFinal Score: {score}/3 - Let's improve this during training!")

# Python Basics Crash Course with Banking Examples
print("\n** Python Basics Workshop **")
# Variables
bank_name = "MegaBank International"
established = 1999
assets = 15_000_000_000  # Using Python 3.6+ underscore notation

print(f"\nWelcome to {bank_name}, est. {established}")
print(f"Total assets: ${assets:,.2f}")

# Conditional Logic Challenge
customer_age = int(input("\nEnter customer age: "))
if customer_age >= 18:
    print("✅ Eligible for banking services")
else:
    print("⛔ Requires guardian approval")

# Lists and Loops
print("\n** Customer Account Generator **")
accounts = []
for i in range(3):
    name = input(f"Enter name for account {i+1}: ")
    balance = float(input("Initial deposit: $"))
    accounts.append((name, balance))

print("\nGenerated Accounts:")
for name, bal in accounts:
    print(f"{name}: ${bal:.2f}")

# ========================
# Week 2: NumPy/Pandas & Data Generation
# ========================
import numpy as np
import pandas as pd
from faker import Faker
import random

print("\n=== WEEK 2: Data Manipulation & Real Banking Data ===")

# Realistic Banking Data Generator
fake = Faker()
np.random.seed(42)

def generate_bank_data(num_customers=500):
    data = []
    for _ in range(num_customers):
        data.append({
            'customer_id': fake.uuid4()[:8],
            'name': fake.name(),
            'age': np.random.randint(18, 90),
            'account_type': random.choice(['Checking', 'Savings', 'CD', 'IRA']),
            'balance': np.random.lognormal(mean=5, sigma=1.5),
            'credit_score': np.random.normal(650, 100),
            'last_transaction_days': np.random.poisson(14),
            'state': fake.state_abbr()
        })
    return pd.DataFrame(data)

# Generate dataset
print("\n🛠 Generating realistic banking dataset...")
bank_df = generate_bank_data()
print("✅ Successfully created dataset with:")
print(f"- {len(bank_df)} customers")
print(f"- Columns: {list(bank_df.columns)}")

# Data Exploration Challenge
print("\n** Data Detective Challenge **")
print("First 5 records:")
print(bank_df.head())
print("\nDataset info:")
print(bank_df.info())

# Interactive Task
search_id = input("\nEnter customer ID to find: ").strip()
result = bank_df[bank_df['customer_id'] == search_id]
if not result.empty:
    print("🔍 Customer Found!")
    print(result)
else:
    print("❌ No customer found - Try again!")

# ========================
# Week 3: Data Cleaning & Preparation
# ========================
print("\n=== WEEK 3: Data Cleaning & Quality Control ===")

# Introduce realistic data issues
bank_df.loc[10:15, 'balance'] = np.nan
bank_df.loc[20:25, 'credit_score'] = -999
duplicates = bank_df.sample(5)
bank_df = pd.concat([bank_df, duplicates])

print("""\n⚠️ ALERT: Data quality issues detected!
Your tasks:
1. Find missing values
2. Handle invalid credit scores
3. Remove duplicates""")

# Cleaning Process
print("\n** Data Cleaning Workshop **")
print("Missing values before:", bank_df.isna().sum())

# Handle missing balances
bank_df['balance'] = bank_df['balance'].fillna(bank_df['balance'].median())

# Fix invalid credit scores
bank_df['credit_score'] = bank_df['credit_score'].apply(
    lambda x: x if x >= 300 else np.nan)
bank_df['credit_score'] = bank_df['credit_score'].fillna(
    bank_df['credit_score'].median())

# Remove duplicates
bank_df = bank_df.drop_duplicates()

print("\nMissing values after:", bank_df.isna().sum())

# Interactive Challenge
def data_quality_audit(df):
    issues = 0
    if df['balance'].min() < 0:
        print("🚨 Negative balances found!")
        issues +=1
    if df['credit_score'].max() > 850:
        print("🚨 Impossible credit scores!")
        issues +=1
    return issues

print("\n** Data Quality Audit **")
issues_found = data_quality_audit(bank_df)
print(f"Total issues found: {issues_found}")

# ========================
# Week 4: Basic Visualization
# ========================
import matplotlib.pyplot as plt

print("\n=== WEEK 4: Data Visualization Basics ===")

# Account Distribution
print("\n** Account Type Distribution **")
type_counts = bank_df['account_type'].value_counts()
plt.figure(figsize=(10,6))
type_counts.plot(kind='bar', color=['blue', 'green', 'purple', 'orange'])
plt.title('MegaBank Account Types Distribution')
plt.xlabel('Account Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()

# Interactive Visualization Challenge
print("\n💡 Visualization Challenge: Create balance distribution plot")
choice = input("Choose plot type (histogram/boxplot): ").lower()

plt.figure(figsize=(10,6))
if choice == 'histogram':
    plt.hist(bank_df['balance'], bins=30, edgecolor='black')
    plt.title('Customer Balance Distribution')
elif choice == 'boxplot':
    plt.boxplot(bank_df['balance'])
    plt.title('Balance Distribution Boxplot')
else:
    print("Invalid choice - showing both!")
    plt.subplot(1,2,1)
    plt.hist(bank_df['balance'], bins=30)
    plt.subplot(1,2,2)
    plt.boxplot(bank_df['balance'])
    
plt.show()

# ========================
# Week 5: Advanced EDA
# ========================
print("\n=== WEEK 5: Exploratory Data Analysis ===")

# Correlation Analysis
print("\n** Feature Relationships **")
numeric_df = bank_df.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10,8))
plt.imshow(corr_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Age Segmentation Analysis
bank_df['age_group'] = pd.cut(bank_df['age'],
                             bins=[18, 30, 45, 60, 100],
                             labels=['18-29', '30-44', '45-59', '60+'])

age_analysis = bank_df.groupby('age_group').agg({
    'balance': 'median',
    'credit_score': 'mean'
}).reset_index()

print("\nAge Group Analysis:")
print(age_analysis)

# Interactive Challenge
print("\n🔎 Investigate Relationships:")
x_var = input("Choose X variable (age/balance/credit_score): ")
y_var = input("Choose Y variable (age/balance/credit_score): ")

plt.figure(figsize=(10,6))
plt.scatter(bank_df[x_var], bank_df[y_var], alpha=0.5)
plt.title(f'{x_var} vs {y_var} Relationship')
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.show()

# ========================
# Week 6: Final Project
# ========================
print("\n=== WEEK 6: Comprehensive Banking Analysis ===")

print("""\nFINAL MISSION: 
Analyze our customer dataset and present key findings to MegaBank executives!
Your tasks:
1. Generate fresh data
2. Clean and prepare data
3. Perform full EDA
4. Create visualizations
5. Present insights""")

# Generate new data
final_df = generate_bank_data(1000)

# Comprehensive Cleaning
def clean_data(df):
    # Add all cleaning steps from previous weeks
    df = df.drop_duplicates()
    df['balance'] = df['balance'].fillna(df['balance'].median())
    df['credit_score'] = df['credit_score'].apply(
        lambda x: x if x >= 300 else np.nan)
    df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
    return df

final_df_clean = clean_data(final_df)

# Full Analysis
print("\n** Automated Report Generation **")
report = {
    'total_customers': len(final_df_clean),
    'avg_balance': final_df_clean['balance'].mean(),
    'most_common_account': final_df_clean['account_type'].mode()[0],
    'top_state': final_df_clean['state'].value_counts().index[0]
}

print("\n=== MegaBank Customer Report ===")
for k,v in report.items():
    print(f"{k.replace('_', ' ').title()}: {v}")

# Visualization Suite
fig, ax = plt.subplots(2,2, figsize=(15,10))
ax[0,0].hist(final_df_clean['balance'], bins=30)
ax[0,0].set_title('Balance Distribution')

final_df_clean['account_type'].value_counts().plot.pie(ax=ax[0,1])

final_df_clean.groupby('age_group')['credit_score'].mean().plot.bar(ax=ax[1,0])
ax[1,0].set_title('Credit Score by Age Group')

ax[1,1].scatter(final_df_clean['age'], final_df_clean['balance'], alpha=0.3)
ax[1,1].set_title('Age vs Balance')

plt.tight_layout()
plt.show()

print("""\n🎉 Congratulations! You've completed the 6-week training!
Now go forth and make data-driven banking decisions!""")

      
# -*- coding: utf-8 -*-
"""
Ultimate Banking Data Science Bootcamp - 8 Weeks to Mastery
Features:
- Real-world Banking Scenarios
- Interactive Coding Challenges
- Statistical Foundations
- Gamified Learning System
- Progressive Complexity
"""

# ================ PRE-SETUP ================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from faker import Faker
import ipywidgets as widgets
from IPython.display import display, clear_output
import random

# Initialize gamification system
student_profile = {
    'level': 1,
    'points': 0,
    'badges': [],
    'data_scientist': False
}

def update_progress(points):
    student_profile['points'] += points
    if student_profile['points'] > 100 and 1 not in student_profile['badges']:
        student_profile['badges'].append(1)
        print("🎖️ New Badge: Data Novice!")
    if student_profile['points'] > 300 and 2 not in student_profile['badges']:
        student_profile['badges'].append(2)
        print("🎖️ New Badge: Analysis Pro!")
    if student_profile['points'] > 500:
        student_profile['data_scientist'] = True

# ================ WEEK 1 ================
def week1():
    print("""\n=== WEEK 1: Python Essentials for Banking ===
    Module 1: Banking Variables & Operations
    Module 2: Customer Data Structures
    Module 3: Conditional Banking Logic""")
    
    # Interactive Story
    print("\n🚀 Welcome to GlobalBank's Data Team!")
    print("""Your first task: Help analyze customer accounts
    to prevent fraud and improve services!""")
    
    # Banking Variables Challenge
    print("\n💻 Coding Challenge 1: Account Manager")
    account_name = input("Enter account holder name: ")
    account_balance = float(input("Enter opening balance: $"))
    interest_rate = 2.5  # APR
    
    # Balance Projection
    print("\n🔮 5-Year Balance Projection:")
    for year in range(1, 6):
        account_balance *= (1 + interest_rate/100)
        print(f"Year {year}: ${account_balance:.2f}")
    
    # Data Structures for Banking
    print("\n📚 Customer Database Creation")
    customers = []
    for _ in range(3):
        name = input("Enter customer name: ")
        age = int(input("Enter age: "))
        customers.append({'name': name, 'age': age})
    
    # Fraud Detection Logic
    print("\n🔍 Fraud Alert System")
    transaction_amount = float(input("Enter transaction amount: $"))
    if transaction_amount > 10000:
        print("🚨 Alert: Large transaction requires verification!")
    else:
        print("✅ Transaction approved")
    
    update_progress(25)

# ================ WEEK 2 ================ 
def week2():
    print("""\n=== WEEK 2: Banking Data Generation & Pandas ===
    Module 1: Realistic Data Simulation
    Module 2: Pandas Operations
    Module 3: Data Quality Inspection""")
    
    # Enhanced Data Generator
    def create_bank_data(records=500):
        fake = Faker()
        np.random.seed(42)
        
        data = []
        for _ in range(records):
            data.append({
                'cust_id': fake.uuid4()[:8],
                'name': fake.name(),
                'age': np.random.randint(18, 90),
                'account_type': random.choice(['Checking', 'Savings', 'CD', 'IRA', 'Business']),
                'balance': np.abs(np.random.lognormal(mean=6, sigma=1.2)),
                'credit_score': np.clip(np.random.normal(650, 100), 300, 850),
                'last_login': np.random.poisson(14),
                'state': fake.state_abbr(),
                'loan_approved': random.choices([True, False], weights=[0.3, 0.7])[0],
                'fraud_risk': random.choices([0,1], weights=[0.97, 0.03])[0]
            })
        return pd.DataFrame(data)
    
    # Generate Dataset
    bank_df = create_bank_data()
    print("\n📊 Generated Banking Dataset:")
    print(f"Shape: {bank_df.shape}")
    print("Columns:", bank_df.columns.tolist())
    
    # Interactive Exploration
    def data_inspector(column):
        clear_output()
        print(bank_df[column].describe())
        if bank_df[column].dtype in ['int64', 'float64']:
            plt.hist(bank_df[column], bins=20)
            plt.show()
            
    columns = widgets.Dropdown(options=bank_df.columns)
    widgets.interact(data_inspector, column=columns)
    
    # Data Cleaning Challenge
    print("\n🔧 Data Quality Challenge")
    bank_df.loc[10:15, 'balance'] = np.nan
    bank_df.loc[20:25, 'credit_score'] = -999
    print("Find and fix data quality issues!")
    
    update_progress(50)

# ================ WEEK 3 ================
def week3():
    print("""\n=== WEEK 3: Statistical Foundations ===
    Module 1: Descriptive Statistics
    Module 2: Probability Distributions
    Module 3: Hypothesis Testing""")
    
    # Generate Financial Data
    np.random.seed(42)
    credit_scores = np.clip(np.random.normal(650, 100, 1000), 300, 850)
    account_balances = np.abs(np.random.lognormal(6, 1.2, 1000))
    
    # Central Tendency Exercise
    print("\n📈 Statistical Analysis Challenge")
    print("Credit Scores Analysis:")
    print(f"Mean: {np.mean(credit_scores):.1f}")
    print(f"Median: {np.median(credit_scores):.1f}")
    print(f"Standard Deviation: {np.std(credit_scores):.1f}")
    
    # Distribution Visualization
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(credit_scores, bins=20)
    plt.title('Credit Score Distribution')
    
    plt.subplot(1,2,2)
    plt.boxplot(account_balances)
    plt.title('Account Balances')
    plt.show()
    
    # Hypothesis Testing
    print("\n🔬 Fraud Detection Hypothesis Test")
    sample = bank_df[bank_df['fraud_risk'] == 1]['balance']
    population = bank_df[bank_df['fraud_risk'] == 0]['balance']
    
    t_stat, p_val = stats.ttest_ind(sample, population)
    print(f"P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("✅ Significant difference in fraud account balances")
    else:
        print("❌ No significant difference found")
    
    update_progress(75)

# ================ WEEK 4 ================
def week4():
    print("""\n=== WEEK 4: Advanced Pandas & Cleaning ===
    Module 1: Data Transformations
    Module 2: Group Analysis
    Module 3: Merging Datasets""")
    
    # Create Transaction Data
    transaction_data = pd.DataFrame({
        'cust_id': np.random.choice(bank_df['cust_id'], 
        'transaction_date': pd.date_range('2023-01-01', periods=500),
        'amount': np.abs(np.random.normal(100, 50, 500))
    })
    
    # Merge Challenge
    print("\n🔗 Data Integration Challenge")
    merged_df = pd.merge(bank_df, transaction_data, on='cust_id')
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Group Analysis
    print("\n📊 Account Type Analysis")
    account_stats = merged_df.groupby('account_type').agg({
        'balance': ['mean', 'median'],
        'amount': ['count', 'mean']
    })
    print(account_stats)
    
    # Time Analysis
    merged_df['transaction_month'] = merged_df['transaction_date'].dt.month
    monthly_stats = merged_df.groupby('transaction_month')['amount'].sum()
    monthly_stats.plot(kind='bar')
    plt.title('Monthly Transaction Volume')
    plt.show()
    
    update_progress(100)

# ================ WEEK 5 ================
def week5():
    print("""\n=== WEEK 5: Data Visualization Mastery ===
    Module 1: Matplotlib Customization
    Module 2: Banking Dashboards
    Module 3: Interactive Visuals""")
    
    # Create Banking Dashboard
    plt.figure(figsize=(15,10))
    
    plt.subplot(2,2,1)
    bank_df['account_type'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Account Types')
    
    plt.subplot(2,2,2)
    plt.scatter(bank_df['age'], bank_df['balance'], alpha=0.5)
    plt.title('Age vs Balance')
    
    plt.subplot(2,2,3)
    pd.crosstab(bank_df['loan_approved'], bank_df['account_type']).plot.bar(stacked=True)
    plt.title('Loan Approvals by Account Type')
    
    plt.subplot(2,2,4)
    plt.hist(bank_df['credit_score'], bins=20)
    plt.title('Credit Score Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Interactive Plot Challenge
    def update_plot(x_axis, y_axis):
        plt.figure(figsize=(10,6))
        plt.scatter(bank_df[x_axis], bank_df[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()
    
    widgets.interact(update_plot,
                    x_axis=['age', 'balance', 'credit_score'],
                    y_axis=['age', 'balance', 'credit_score'])
    
    update_progress(125)

# ================ WEEK 6 ================
def week6():
    print("""\n=== WEEK 6: EDA Foundations ===
    Module 1: Correlation Analysis
    Module 2: Feature Engineering
    Module 3: Outlier Detection""")
    
    # Correlation Heatmap
    numeric_df = bank_df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12,8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=range(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=45)
    plt.yticks(ticks=range(len(corr_matrix.columns)), labels=corr_matrix.columns)
    plt.title('Banking Features Correlation')
    plt.show()
    
    # Outlier Detection Challenge
    print("\n🔎 Outlier Investigation")
    z_scores = np.abs(stats.zscore(numeric_df))
    outliers = (z_scores > 3).any(axis=1)
    print(f"Found {outliers.sum()} potential outliers")
    
    # Feature Engineering
    bank_df['balance_to_income_ratio'] = bank_df['balance'] / bank_df['credit_score']
    bank_df['age_group'] = pd.cut(bank_df['age'],
                                 bins=[18, 30, 45, 60, 100],
                                 labels=['Young', 'Mid-Career', 'Pre-Retirement', 'Retired'])
    
    update_progress(150)

# ================ WEEK 7 ================
def week7():
    print("""\n=== WEEK 7: Advanced Statistics ===
    Module 1: Regression Analysis
    Module 2: Confidence Intervals
    Module 3: Bayesian Statistics""")
    
    # Linear Regression
    from sklearn.linear_model import LinearRegression
    
    X = bank_df[['credit_score', 'age']]
    y = bank_df['balance']
    
    model = LinearRegression().fit(X, y)
    print(f"\n📈 Balance Prediction Model R²: {model.score(X,y):.2f}")
    
    # Confidence Intervals
    mean_balance = np.mean(bank_df['balance'])
    sem = stats.sem(bank_df['balance'])
    ci = stats.t.interval(0.95, len(bank_df['balance'])-1, mean_balance, sem)
    
    print(f"\n95% Confidence Interval for Balances:")
    print(f"${ci[0]:.2f} to ${ci[1]:.2f}")
    
    # Bayesian A/B Test
    from scipy.stats import beta
    
    # Loan approval rates
    group_a = [120, 380]  # approved, denied
    group_b = [150, 350]
    
    prior = beta(1, 1)
    posterior_a = beta(prior.args[0] + group_a[0], prior.args[1] + group_a[1])
    posterior_b = beta(prior.args[0] + group_b[0], prior.args[1] + group_b[1])
    
    samples = 100000
    a_samples = posterior_a.rvs(samples)
    b_samples = posterior_b.rvs(samples)
    print(f"\nProbability B better than A: {(b_samples > a_samples).mean():.1%}")
    
    update_progress(175)

# ================ WEEK 8 ================
def week8():
    print("""\n=== WEEK 8: Capstone Project ===
    Module 1: Full Data Analysis
    Module 2: Executive Reporting
    Module 3: Presentation Skills""")
    
    # Final Project
    print("""\n🎓 FINAL CHALLENGE:
    Analyze GlobalBank's customer data and prepare
    a report for executive leadership!""")
    
    # Comprehensive Analysis
    final_df = create_bank_data(5000)
    
    # Full EDA Process
    def complete_analysis(df):
        # Data Cleaning
        df = df.drop_duplicates()
        df['credit_score'] = df['credit_score'].clip(300, 850)
        df['balance'] = df['balance'].fillna(df['balance'].median())
        
        # Feature Engineering
        df['risk_category'] = pd.qcut(df['fraud_risk'], 3, labels=['Low', 'Medium', 'High'])
        
        # Statistical Analysis
        report = {
            'total_customers': len(df),
            'avg_balance': df['balance'].mean(),
            'loan_approval_rate': df['loan_approved'].mean(),
            'fraud_rate': df['fraud_risk'].mean(),
            'top_state': df['state'].mode()[0]
        }
        
        # Visualizations
        fig, ax = plt.subplots(2,2, figsize=(15,10))
        df['account_type'].value_counts().plot.pie(ax=ax[0,0])
        pd.crosstab(df['age_group'], df['risk_category']).plot.bar(ax=ax[0,1])
        ax[1,0].scatter(df['credit_score'], df['balance'])
        df['transaction_month'].value_counts().sort_index().plot(ax=ax[1,1])
        
        return report, fig
    
    # Generate Report
    final_report, final_fig = complete_analysis(final_df)
    print("\n=== Executive Summary ===")
    for k,v in final_report.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
    final_fig.show()
    
    print("\n🎉 CONGRATULATIONS! You've completed the bootcamp!")
    print(student_profile)
    update_progress(200)

# ================ MAIN EXECUTION ================
if __name__ == "__main__":
    weeks = [week1, week2, week3, week4, week5, week6, week7, week8]
    
    print("""Welcome to GlobalBank Data Science Bootcamp!
    Complete 8 weeks of training to become a banking data expert!""")
    
    for week_num, week_func in enumerate(weeks, 1):
        input(f"\nPress Enter to start Week {week_num}...")
        week_func()


Banking Operations Automation System
Key Features:
- Interest Calculation
- Fraud Detection
- Customer Reporting
- Portfolio Analysis


import pandas as pd
import numpy as np
from faker import Faker
import matplotlib.pyplot as plt

# Generate realistic banking data
fake = Faker()
np.random.seed(42)

def generate_bank_data(num_customers=100):
    """Simulate customer accounts with transactions"""
    accounts = []
    for _ in range(num_customers):
        account = {
            'cust_id': fake.uuid4()[:8],
            'name': fake.name(),
            'account_type': np.random.choice(['Savings', 'Checking', 'CD'], p=[0.6, 0.3, 0.1]),
            'balance': np.random.lognormal(mean=6, sigma=0.5),
            'transactions': [
                {'date': fake.date_this_year(), 
                 'amount': np.random.normal(500, 200),
                 'type': np.random.choice(['Deposit', 'Withdrawal']}
                for _ in range(np.random.randint(5, 15))]
        }
        accounts.append(account)
    return accounts

# Generate dataset
bank_data = generate_bank_data()

# ======================
# Core Banking Functions
# ======================

def calculate_interest(account, days=30):
    """Calculate monthly interest with compounding"""
    rates = {'Savings': 0.02, 'Checking': 0.01, 'CD': 0.03}
    daily_rate = rates[account['account_type']] / 365
    return account['balance'] * (1 + daily_rate)**days - account['balance']

def detect_fraud(transactions):
    """Identify suspicious transaction patterns"""
    alerts = []
    for i, t in enumerate(transactions):
        # Rule 1: Large transaction > $2000
        if abs(t['amount']) > 2000:
            alerts.append(f"Large transaction: ${t['amount']:.2f} on {t['date']}")
        
        # Rule 2: 3+ consecutive withdrawals
        if i >= 2 and all(transactions[j]['type'] == 'Withdrawal' for j in [i-2, i-1, i]):
            alerts.append(f"Multiple withdrawals on {t['date']}")
    return alerts

def generate_customer_report(account):
    """Create personalized customer statement"""
    report = f"\n=== {account['name']} Account Statement ==="
    report += f"\nAccount Type: {account['account_type']}"
    report += f"\nCurrent Balance: ${account['balance']:.2f}"
    
    # Calculate interest using our function
    interest = calculate_interest(account)
    report += f"\nProjected Monthly Interest: ${interest:.2f}"
    
    # Fraud detection using our function
    fraud_alerts = detect_fraud(account['transactions'])
    if fraud_alerts:
        report += "\n\n⚠️ Security Alerts:"
        for alert in fraud_alerts:
            report += f"\n• {alert}"
    
    return report

# ======================
# Bulk Processing (Loops)
# ======================

# Process all accounts using loops and functions
total_interest = 0
fraud_cases = 0

print("🔍 Analyzing customer accounts...")
for account in bank_data:
    # Update balances with interest
    interest = calculate_interest(account)
    account['balance'] += interest
    total_interest += interest
    
    # Check for fraud
    if detect_fraud(account['transactions']):
        fraud_cases += 1
    
    # Generate customer report
    print(generate_customer_report(account))

# ======================
# Portfolio Analysis
# ======================

# Create DataFrame for analysis
df = pd.DataFrame([{
    'Customer': acc['name'],
    'Account Type': acc['account_type'],
    'Balance': acc['balance'],
    'Transactions': len(acc['transactions'])
} for acc in bank_data])

# Aggregate statistics using pandas
portfolio_stats = df.groupby('Account Type').agg({
    'Balance': ['mean', 'sum'],
    'Transactions': 'mean'
})

print("\n📊 Portfolio Summary:")
print(portfolio_stats)

# Visualize account distributions
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
df['Account Type'].value_counts().plot.pie(autopct='%1.1f%%')

plt.subplot(1,2,2)
plt.scatter(df['Transactions'], df['Balance'])
plt.xlabel('Number of Transactions')
plt.ylabel('Account Balance')
plt.title('Transaction Activity vs Balance')

plt.tight_layout()
plt.show()

# Final report
print(f"\n💹 Total Interest Projected: ${total_interest:.2f}")
print(f"🚨 Potential Fraud Cases Detected: {fraud_cases}")

"""
Python Functions & List Comprehensions Masterclass
with Banking Applications
"""

# ======================
# 1. Basic Functions
# ======================

def calculate_interest(principal, rate, years):
    """Calculate compound interest for a loan"""
    return principal * (1 + rate/100) ** years - principal

# Banking usage
loan_amount = 100000
print(f"5 Year Interest: ${calculate_interest(loan_amount, 5, 5):.2f}")

# ======================
# 2. Function Parameters
# ======================

# Default parameters
def create_account(name, acc_type="Savings", min_balance=100):
    return {"name": name, "type": acc_type, "balance": min_balance}

# *args example
def process_transactions(account, *transactions):
    for amount in transactions:
        account['balance'] += amount

# **kwargs example
def customer_profile(**details):
    return {k:v for k,v in details.items()}

# Banking usage
new_account = create_account("John Doe")
process_transactions(new_account, 500, -200, 1500)
print(f"Updated Balance: ${new_account['balance']}")

# ======================
# 3. Lambda Functions
# ======================

# Simple fee calculation
overdraft_fee = lambda balance: 35 if balance < 0 else 0
print(f"Fee charged: ${overdraft_fee(-500)}")

# Sorting transactions
transactions = [150, -200, 500, -50]
sorted_trans = sorted(transactions, key=lambda x: abs(x), reverse=True)
print("Large transactions first:", sorted_trans)

# ======================
# 4. List Comprehensions
# ======================

# Basic comprehension
daily_transactions = [150, -200, 500, -50, 1000]
positive_transactions = [t for t in daily_transactions if t > 0]
print("Deposits:", positive_transactions)

# Complex transformation
currencies = [('USD', 150), ('EUR', 200), ('GBP', 300)]
exchange_rates = {'USD': 1, 'EUR': 1.2, 'GBP': 1.35}
usd_values = [amt * exchange_rates[curr] for curr, amt in currencies]
print("USD equivalents:", usd_values)

# Nested comprehension
fraud_patterns = [[150, 200, 5000], [300, 450], [20, 5000]]
suspicious = [trans for account in fraud_patterns for trans in account if trans > 2000]
print("Suspicious transactions:", suspicious)

# ======================
# 5. Advanced Functions
# ======================

# Closure example
def transaction_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

# Banking usage
atm_counter = transaction_counter()
print("Transaction count:", atm_counter(), atm_counter())

# Decorator example
def audit_trail(func):
    def wrapper(*args, **kwargs):
        print(f"Audit: Calling {func.__name__} with {args} {kwargs}")
        return func(*args, **kwargs)
    return wrapper

@audit_trail
def approve_loan(amount):
    return amount <= 1000000

print("Loan approval:", approve_loan(500000))

# ======================
# 6. Banking Applications
# ======================

# Fraud detection pipeline
transactions = [150, -200, 5000, -50, 2500, -300]

# Pipeline using list comprehensions and lambda
detect_fraud = lambda x: x if abs(x) > 2000 else None
fraud_candidates = [detect_fraud(t) for t in transactions]
fraud_candidates = [t for t in fraud_candidates if t is not None]
print("Potential fraud transactions:", fraud_candidates)

# Customer report generation
customers = [
    {'name': 'Alice', 'balance': 1500},
    {'name': 'Bob', 'balance': -200},
    {'name': 'Charlie', 'balance': 5000}
]

report = [f"{c['name']}: {'OK' if c['balance'] >=0 else 'OVERDRAWN'}" 
          for c in customers]
print("\nCustomer Status:")
print('\n'.join(report))

# ======================
# 7. Best Practices
# ======================
"""
Key Considerations:
1. Use regular functions for complex logic
2. Reserve lambdas for simple, single-expression operations
3. Prefer list comprehensions over map/filter for readability
4. Use type hints for banking calculations
5. Maintain function purity (no side effects) when possible
6. Document financial formulas clearly
"""

# Typed function example
def calculate_emi(principal: float, rate: float, months: int) -> float:
    """
    Calculate Equated Monthly Installment (EMI)
    
    Args:
        principal: Loan amount
        rate: Annual interest rate percentage
        months: Loan tenure in months
        
    Returns:
        Monthly payment amount
    """
    monthly_rate = rate / 1200
    return (principal * monthly_rate * (1 + monthly_rate)**months) / \
           ((1 + monthly_rate)**months - 1)

print(f"\nEMI for $100k loan: ${calculate_emi(100000, 7.5, 60):.2f}/month")
                 
