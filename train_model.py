import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# قراءة البيانات
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# تحويل القيم النصية إلى أرقام (التصنيف)
df_encoded = pd.get_dummies(df.drop("Attrition", axis=1))  # حذف العمود الهدف مؤقتًا وتحويل الباقي
y = df["Attrition"].map({"Yes": 1, "No": 0})  # تحويل الهدف إلى 1 و 0

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = XGBClassifier()
model.fit(X_train, y_train)

# حفظ النموذج
joblib.dump(model, "xgb_model.pkl")

# حفظ أسماء الأعمدة المستخدمة في التدريب
joblib.dump(X_train.columns.tolist(), "model_features.pkl")

print("تم تدريب النموذج وحفظه بنجاح.")
