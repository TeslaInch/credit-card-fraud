import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load & use 10% of the data only
df = pd.read_csv('creditcard_small.csv')
df = df.sample(frac=0.1, random_state=42)  # FAST

# Preprocess the data
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df.drop('Time', axis=1, inplace=True)
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# Train Random Forest FAST
model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Save
joblib.dump(model, 'models/model.pkl')
print("✅ Model saved.")
