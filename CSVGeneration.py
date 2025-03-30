import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Créditos: Este trabajo fue adaptado y editado para este caso con base en el dataset de Kaggle:
# https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval

# Función para generar un archivo CSV con datos sintéticos personalizados
def generate_custom_csv(approved_count, rejected_count, include_noise=True, include_randomness=True, output_file_name="custom_data.csv"):
    """
    Genera un archivo CSV con datos sintéticos para simulación de aprobación de préstamos.

    Parámetros:
    - approved_count: Número de préstamos aprobados.
    - rejected_count: Número de préstamos rechazados.
    - include_noise: Si se incluye ruido en los datos (default: True).
    - include_randomness: Si se permite aleatoriedad en los datos (default: True).
    - output_file_name: Nombre del archivo de salida (default: "custom_data.csv").
    """
    total_samples = approved_count + rejected_count
    np.random.seed(42 if not include_randomness else None)  # Control de aleatoriedad

    # Generar datos sintéticos
    age = np.random.normal(40, 12, total_samples).clip(18, 80).astype(int)
    experience = (age - 18 - np.random.normal(4, 2, total_samples).clip(0)).clip(0).astype(int)
    education_level = np.random.choice(['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'], total_samples, p=[0.3, 0.2, 0.3, 0.15, 0.05])
    annual_income = np.random.lognormal(10.5, 0.6, total_samples).clip(15000, 300000).astype(int)
    credit_score = (300 + 300 * np.random.beta(5, 1.5, total_samples)).clip(300, 850).astype(int)
    employment_status = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], total_samples, p=[0.7, 0.2, 0.1])
    application_dates = [datetime(2018, 1, 1) + timedelta(days=i) for i in range(total_samples)]

    data = {
        'ApplicationDate': application_dates,
        'Age': age,
        'AnnualIncome': annual_income,
        'CreditScore': credit_score,
        'EmploymentStatus': employment_status,
        'EducationLevel': education_level,
        'Experience': experience,
        'LoanAmount': np.random.lognormal(10, 0.5, total_samples).astype(int),
        'LoanDuration': np.random.choice([12, 24, 36, 48, 60, 72, 84, 96, 108, 120], total_samples, p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.025, 0.025]),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], total_samples, p=[0.3, 0.5, 0.15, 0.05]),
        'NumberOfDependents': np.random.choice([0, 1, 2, 3, 4, 5], total_samples, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),
        'HomeOwnershipStatus': np.random.choice(['Own', 'Rent', 'Mortgage', 'Other'], total_samples, p=[0.2, 0.3, 0.4, 0.1]),
        'MonthlyDebtPayments': np.random.lognormal(6, 0.5, total_samples).astype(int),
        'CreditCardUtilizationRate': np.random.beta(2, 5, total_samples),
        'NumberOfOpenCreditLines': np.random.poisson(3, total_samples).clip(0, 15).astype(int),
        'NumberOfCreditInquiries': np.random.poisson(1, total_samples).clip(0, 10).astype(int),
        'DebtToIncomeRatio': np.random.beta(2, 5, total_samples),
        'BankruptcyHistory': np.random.choice([0, 1], total_samples, p=[0.95, 0.05]),
        'LoanPurpose': np.random.choice(['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other'], total_samples, p=[0.3, 0.2, 0.15, 0.25, 0.1]),
        'PreviousLoanDefaults': np.random.choice([0, 1], total_samples, p=[0.9, 0.1]),
        'PaymentHistory': np.random.poisson(24, total_samples).clip(0, 60).astype(int),
        'LengthOfCreditHistory': np.random.randint(1, 30, total_samples),
        'SavingsAccountBalance': np.random.lognormal(8, 1, total_samples).astype(int),
        'CheckingAccountBalance': np.random.lognormal(7, 1, total_samples).astype(int),
        'TotalAssets': np.random.lognormal(11, 1, total_samples).astype(int),
        'TotalLiabilities': np.random.lognormal(10, 1, total_samples).astype(int),
        'MonthlyIncome': annual_income / 12,
        'UtilityBillsPaymentHistory': np.random.beta(8, 2, total_samples),
        'JobTenure': np.random.poisson(5, total_samples).clip(0, 40).astype(int),
    }

    df = pd.DataFrame(data)
    df['TotalAssets'] = np.maximum(df['TotalAssets'], df['SavingsAccountBalance'] + df['CheckingAccountBalance'])
    min_net_worth = 1000
    df['NetWorth'] = np.maximum(df['TotalAssets'] - df['TotalLiabilities'], min_net_worth)
    df['BaseInterestRate'] = 0.03 + (850 - df['CreditScore']) / 2000 + df['LoanAmount'] / 1000000 + df['LoanDuration'] / 1200
    df['InterestRate'] = df['BaseInterestRate'] * (1 + np.random.normal(0, 0.1, total_samples)).clip(0.8, 1.2)
    df['MonthlyLoanPayment'] = (df['LoanAmount'] * (df['InterestRate'] / 12)) / (1 - (1 + df['InterestRate'] / 12)**(-df['LoanDuration']))
    df['TotalDebtToIncomeRatio'] = (df['MonthlyDebtPayments'] + df['MonthlyLoanPayment']) / df['MonthlyIncome']

    # Regla de aprobación de préstamos
    def loan_approval_rule(row):
        score = 0
        score += (row['CreditScore'] - 600) / 250
        score += (100000 - row['AnnualIncome']) / 100000
        score += (row['TotalDebtToIncomeRatio'] - 0.4) * 2
        score += (row['LoanAmount'] - 10000) / 90000
        score += (row['InterestRate'] - 0.05) * 10
        score += 0.5 if row['BankruptcyHistory'] == 1 else 0
        score += 0.3 if row['PreviousLoanDefaults'] == 1 else 0
        score += 0.2 if row['EmploymentStatus'] == 'Unemployed' else 0
        score -= 0.1 if row['HomeOwnershipStatus'] in ['Own', 'Mortgage'] else 0
        score -= row['PaymentHistory'] / 120
        score -= row['LengthOfCreditHistory'] / 60
        score -= row['NetWorth'] / 500000
        score += abs(row['Age'] - 40) / 100
        score -= row['Experience'] / 200
        edu_score = {'High School': 0.2, 'Associate': 0.1, 'Bachelor': 0, 'Master': -0.1, 'Doctorate': -0.2}
        score += edu_score[row['EducationLevel']]
        month = row['ApplicationDate'].month
        score -= 0.1 if 3 <= month <= 8 else 0
        score += np.random.normal(0, 0.1)
        return 1 if score < 1 else 0

    df.insert(0, 'LoanApproved', df.apply(loan_approval_rule, axis=1))
    
    # Ajustar la cantidad de aprobados y rechazados si es necesario
    approved_indices = df[df['LoanApproved'] == 1].index
    rejected_indices = df[df['LoanApproved'] == 0].index
    approved_deficit = max(0, approved_count - len(approved_indices))
    rejected_deficit = max(0, rejected_count - len(rejected_indices))

    # Eliminar filas aleatorias si hay exceso de aprobados o rechazados
    if len(approved_indices) > approved_count:
        drop_indices = np.random.choice(approved_indices, len(approved_indices) - approved_count, replace=False)
        df.drop(drop_indices, inplace=True)
    if len(rejected_indices) > rejected_count:
        drop_indices = np.random.choice(rejected_indices, len(rejected_indices) - rejected_count, replace=False)
        df.drop(drop_indices, inplace=True)

    # Generar nuevos datos para cumplir con la cantidad deseada de aprobados y rechazados
    while approved_deficit > 0 or rejected_deficit > 0:
        new_age = np.random.normal(40, 12, 1).clip(18, 80).astype(int)
        new_experience = (new_age - 18 - np.random.normal(4, 2, 1).clip(0)).clip(0).astype(int)
        new_education_level = np.random.choice(['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'], 1, p=[0.3, 0.2, 0.3, 0.15, 0.05])
        new_annual_income = np.random.lognormal(10.5, 0.6, 1).clip(15000, 300000).astype(int)
        new_credit_score = (300 + 300 * np.random.beta(5, 1.5, 1)).clip(300, 850).astype(int)
        new_employment_status = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], 1, p=[0.7, 0.2, 0.1])
        new_application_date = [datetime(2018, 1, 1) + timedelta(days=np.random.randint(0, 365))]

        new_data = {
            'ApplicationDate': new_application_date,
            'Age': new_age,
            'AnnualIncome': new_annual_income,
            'CreditScore': new_credit_score,
            'EmploymentStatus': new_employment_status,
            'EducationLevel': new_education_level,
            'Experience': new_experience,
            'LoanAmount': np.random.lognormal(10, 0.5, 1).astype(int),
            'LoanDuration': np.random.choice([12, 24, 36, 48, 60, 72, 84, 96, 108, 120], 1, p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.025, 0.025]),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 1, p=[0.3, 0.5, 0.15, 0.05]),
            'NumberOfDependents': np.random.choice([0, 1, 2, 3, 4, 5], 1, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),
            'HomeOwnershipStatus': np.random.choice(['Own', 'Rent', 'Mortgage', 'Other'], 1, p=[0.2, 0.3, 0.4, 0.1]),
            'MonthlyDebtPayments': np.random.lognormal(6, 0.5, 1).astype(int),
            'CreditCardUtilizationRate': np.random.beta(2, 5, 1),
            'NumberOfOpenCreditLines': np.random.poisson(3, 1).clip(0, 15).astype(int),
            'NumberOfCreditInquiries': np.random.poisson(1, 1).clip(0, 10).astype(int),
            'DebtToIncomeRatio': np.random.beta(2, 5, 1),
            'BankruptcyHistory': np.random.choice([0, 1], 1, p=[0.95, 0.05]),
            'LoanPurpose': np.random.choice(['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other'], 1, p=[0.3, 0.2, 0.15, 0.25, 0.1]),
            'PreviousLoanDefaults': np.random.choice([0, 1], 1, p=[0.9, 0.1]),
            'PaymentHistory': np.random.poisson(24, 1).clip(0, 60).astype(int),
            'LengthOfCreditHistory': np.random.randint(1, 30, 1),
            'SavingsAccountBalance': np.random.lognormal(8, 1, 1).astype(int),
            'CheckingAccountBalance': np.random.lognormal(7, 1, 1).astype(int),
            'TotalAssets': np.random.lognormal(11, 1, 1).astype(int),
            'TotalLiabilities': np.random.lognormal(10, 1, 1).astype(int),
            'MonthlyIncome': new_annual_income / 12,
            'UtilityBillsPaymentHistory': np.random.beta(8, 2, 1),
            'JobTenure': np.random.poisson(5, 1).clip(0, 40).astype(int),
        }

        new_row = pd.DataFrame(new_data)
        new_row['TotalAssets'] = np.maximum(new_row['TotalAssets'], new_row['SavingsAccountBalance'] + new_row['CheckingAccountBalance'])
        new_row['NetWorth'] = np.maximum(new_row['TotalAssets'] - new_row['TotalLiabilities'], min_net_worth)
        new_row['BaseInterestRate'] = 0.03 + (850 - new_row['CreditScore']) / 2000 + new_row['LoanAmount'] / 1000000 + new_row['LoanDuration'] / 1200
        new_row['InterestRate'] = new_row['BaseInterestRate'] * (1 + np.random.normal(0, 0.1, 1)).clip(0.8, 1.2)
        new_row['MonthlyLoanPayment'] = (new_row['LoanAmount'] * (new_row['InterestRate'] / 12)) / (1 - (1 + new_row['InterestRate'] / 12)**(-new_row['LoanDuration']))
        new_row['TotalDebtToIncomeRatio'] = (new_row['MonthlyDebtPayments'] + new_row['MonthlyLoanPayment']) / new_row['MonthlyIncome']
        new_row['LoanApproved'] = new_row.apply(loan_approval_rule, axis=1)

        aprobation = new_row['LoanApproved'].values[0]
        if aprobation == 1 and approved_deficit > 0:
            df = pd.concat([df, new_row], ignore_index=True)
            approved_deficit -= 1
        elif aprobation == 0 and rejected_deficit > 0:
            df = pd.concat([df, new_row], ignore_index=True)
            rejected_deficit -= 1

    # Agregar ruido si se solicita
    if include_noise:
        noise_mask = np.random.choice([True, False], len(df), p=[0.01, 0.99])
        df.loc[noise_mask, 'AnnualIncome'] = (df.loc[noise_mask, 'AnnualIncome'] * np.random.uniform(1.5, 2.0, noise_mask.sum())).astype(int)
        low_net_worth_mask = df['NetWorth'] == min_net_worth
        df.loc[low_net_worth_mask, 'NetWorth'] += np.random.randint(0, 10000, size=low_net_worth_mask.sum())

    # Convertir columnas categóricas a numéricas y barajar las filas
    # Esto no era parte del código original, pero se agregó para obtener datos que el modelo pueda trabajar directamente

    df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])
    df['Season'] = df['ApplicationDate'].dt.month % 12 // 3 + 1
    season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['Season'] = df['Season'].map(season_mapping)
    df = pd.get_dummies(df, dtype=int, columns=['EducationLevel', 'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose', 'Season'])
    df.drop(columns=['ApplicationDate'], inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)  # Barajar filas

    # Guardar en archivo CSV
    df.to_csv(output_file_name, index=False)
    print(f"\nDatos sintéticos guardados en '{output_file_name}'")


# Parámetros predeterminados para replicar resultados del informe:
# - Total de datos: 6423
# - Aprobados: 54% (3468)
# - Rechazados: 46% (2955)
# - Ruido: True
# - Aleatoriedad: False

total_datos = 800
aprobados = int(total_datos * 0.54)
rechazados = total_datos - aprobados
ruido = True
aleatorio = False

# Generar nombre del archivo con formato descriptivo
timestamp = datetime.now().strftime("%Y%m%d")
nombre_archivo = f"loan_data_{total_datos}_samples_{'noise' if ruido else 'no_noise'}_{'random' if aleatorio else 'fixed'}_{timestamp}.csv"

if __name__ == "__main__":

    # Generar CSV con los parámetros definidos
    generate_custom_csv(
        approved_count=aprobados,
        rejected_count=rechazados,
        include_noise=ruido,
        include_randomness=aleatorio,
        output_file_name=nombre_archivo
    )

