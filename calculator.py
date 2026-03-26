import streamlit as st

st.title("Simple Calculator")

# input numbers
num1 = st.number_input("First number", value=0.0, format="%f")
num2 = st.number_input("Second number", value=0.0, format="%f")

operation = st.selectbox("Operation", ["+", "-", "*", "/", "%", "**"])

result = None

if st.button("Compute"):
    try:
        if operation == "+":
            result = num1 + num2
        elif operation == "-":
            result = num1 - num2
        elif operation == "*":
            result = num1 * num2
        elif operation == "/":
            result = num1 / num2
        elif operation == "%":
            result = num1 % num2
        elif operation == "**":
            result = num1 ** num2
    except Exception as e:
        st.error(f"Error performing calculation: {e}")

if result is not None:
    st.success(f"Result: {result}")
