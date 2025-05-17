import streamlit as st
import numpy as np

# --- LU Decomposition (Doolittle’s Method) ---
def lu_decomposition(A):
    n = len(A)
    L = np.zeros_like(A, dtype=float)
    U = np.zeros_like(A, dtype=float)

    for i in range(n):
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
        L[i][i] = 1
        for k in range(i+1, n):
            L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]
    
    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i,:i], y[:i])
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y, dtype=float)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
    return x

# --- Streamlit UI ---
st.title("LU Decomposition Calculator (Doolittle's Method)")
st.write("Solve Ax = b using LU Decomposition")

n = st.number_input("Matrix size (n x n)", min_value=2, max_value=10, value=3)

st.write("Enter matrix A (each row comma-separated):")
A = []
for i in range(n):
    row_input = st.text_input(f"Row {i+1}", value=",".join(["1"]*n))
    row = list(map(float, row_input.split(',')))
    if len(row) != n:
        st.error("Each row must have exactly n elements")
    A.append(row)

b_input = st.text_input("Enter vector b (comma-separated)", value="1,1,1")
b = list(map(float, b_input.split(',')))

if st.button("Solve"):
    try:
        A_np = np.array(A)
        b_np = np.array(b)

        L, U = lu_decomposition(A_np)
        y = forward_substitution(L, b_np)
        x = backward_substitution(U, y)

        st.subheader("Results:")
        st.write("Matrix A:")
        st.write(A_np)
        st.write("Lower triangular matrix L:")
        st.write(L)
        st.write("Upper triangular matrix U:")
        st.write(U)
        st.write("Intermediate solution y (from Ly = b):")
        st.write(y)
        st.write("Final solution x (from Ux = y):")
        st.write(x)

    except Exception as e:
        st.error(f"An error occurred: {e}")
