import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

html_temp = """
<div style ="background-color: skyblue; padding: 1px">
<h2 style ="color: black; text-align:center;">Loan Default Visualization </h2>
</div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

df = pd.read_csv("C:\\Users\\Jalpesh Patel\\Downloads\\Loan_default.csv")


df = df.dropna()

df = df.sample(5000)

x = df.drop(['LoanID','Default'], axis=1)
y = df['Default']

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=20, test_size=0.30, stratify=y)

train_x['Education'] = train_x['Education'].map({"High School":0,"Bachelor's":1,"Master's":2,"PhD":3})
train_x['EmploymentType'] = train_x['EmploymentType'].map({"Unemployed":0,"Self-employed":1,"Part-time":2,"Full-time":3})
train_x['MaritalStatus'] = train_x['MaritalStatus'].map({"Single":0,"Divorced":1,"Married":2})
train_x['HasMortgage'] = train_x['HasMortgage'].map({"No":0,"Yes":1})
train_x['HasDependents'] = train_x['HasDependents'].map({"No":0,"Yes":1})
train_x['LoanPurpose'] = train_x['LoanPurpose'].map({"Other":0,"Auto":1,"Business":2,"Education":3,"Home":4})
train_x['HasCoSigner'] = train_x['HasCoSigner'].map({"No":0,"Yes":1})

test_x['Education'] = test_x['Education'].map({"High School":0,"Bachelor's":1,"Master's":2,"PhD":3})
test_x['EmploymentType'] = test_x['EmploymentType'].map({"Unemployed":0,"Self-employed":1,"Part-time":2,"Full-time":3})
test_x['MaritalStatus'] = test_x['MaritalStatus'].map({"Single":0,"Divorced":1,"Married":2})
test_x['HasMortgage'] = test_x['HasMortgage'].map({"No":0,"Yes":1})
test_x['HasDependents'] = test_x['HasDependents'].map({"No":0,"Yes":1})
test_x['LoanPurpose'] = test_x['LoanPurpose'].map({"Other":0,"Auto":1,"Business":2,"Education":3,"Home":4})
test_x['HasCoSigner'] = test_x['HasCoSigner'].map({"No":0,"Yes":1})



from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(train_x, train_y)

feat_importances = pd.Series(model.feature_importances_, index=x.columns)
top_features = feat_importances.nlargest(10)

top_features = top_features[::-1]


top_features_df = pd.DataFrame({'Features': top_features.index, 'Importance': top_features.values})

# Creating a bar chart using Plotly Express
fig = px.bar(top_features_df, x='Importance', y='Features', orientation='h', title='Top 10 Feature Importances')

fig.update_layout(width=550,height=350,xaxis_title='Importance',yaxis_title='Features', title_x=0.2)


df1 = df.select_dtypes(include='number')

df_num = df1.drop(labels=["Default"], axis=1).columns.tolist()

 
fig1 = px.scatter(df, x="LoanTerm", y="DTIRatio", color="Default")

# Highlighted point using plotly.graph_objects
highlighted_point = px.scatter(df[250:251],x="LoanTerm", y="DTIRatio").update_traces(marker=dict(symbol='circle',size=20, color='magenta'),).data[0]

# Add the highlighted point to the scatter plot
plt.figure(figsize=(4, 2))
fig1.add_trace(highlighted_point)
fig1.update_layout(xaxis=dict(dtick=2,) , width=550,height=350, title="Relation between Loan Term and DTI Ratio", title_x=0.2,title_y=0.9,
                   xaxis_title="Loan Term", yaxis_title= "DTI Ratio")

col1, col2 = st.columns(2)
    
with col1:
    st.plotly_chart(fig)

with col2:
    st.plotly_chart(fig1)

col3, col4 = st.columns(2)


with col3:
    Default_0 = df[df['Default'] == 0]
    Default_1 = df[df['Default'] == 1]

    # Plotly figure
    fig = px.histogram(df, x='Age', color='Default', nbins=10, barmode='overlay',
                    labels={'Age': 'Age', 'Default': 'Defaulter'},
                    title='Relation between Age and Defaulter', color_discrete_sequence=['blue', 'red'])

    # Update layout for better readability
    fig.update_layout(xaxis_title_text='Age', yaxis_title_text='Number of Applicant', legend_title_text='Defaulter', bargap=0.2, width=500, height=350,
                      title_x=0.2, title_y=0.95, xaxis=dict(dtick=10,))

    # Show the figure using Streamlit
    st.plotly_chart(fig)

with col4:
    df['Default1']= df['Default'].map({0.0 : "Not Defaulter", 1.0 : "Defaulter"})
    df['Income_Avg'] = df['Income'].mean()
    mean = df['Income'].mean()
    df['Income_Category'] = df['Income'].apply(lambda x: 'Income Below Avg' if x < mean else 'Income Above Avg')
    fig = px.sunburst(df, path=['Income_Category','Default1'],values='Income_Avg' ,color='Age', color_continuous_scale=['red', 'blue'])
    fig.update_layout(width=600,height=350, title="Relation between Average Income and Defaulter", title_x=0.2)
    st.plotly_chart(fig)





























