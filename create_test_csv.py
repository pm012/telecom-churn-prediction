# create_test_csv.py
import pandas as pd
import numpy as np

def create_test_csv():
    """Створення тестового CSV файлу для пакетної обробки"""
    
    # Створення різних сценаріїв клієнтів
    test_data = [
        # Низький ризик
        {
            'id': 1,
            'is_tv_subscriber': 1,
            'is_movie_package_subscriber': 1,
            'subscription_age': 24.0,
            'bill_avg': 30.0,
            'reamining_contract': 12.0,
            'service_failure_count': 0,
            'download_avg': 50.0,
            'upload_avg': 10.0,
            'download_over_limit': 0
        },
        {
            'id': 2,
            'is_tv_subscriber': 1,
            'is_movie_package_subscriber': 0,
            'subscription_age': 18.0,
            'bill_avg': 45.0,
            'reamining_contract': 8.0,
            'service_failure_count': 1,
            'download_avg': 35.0,
            'upload_avg': 8.0,
            'download_over_limit': 0
        },
        # Середній ризик
        {
            'id': 3,
            'is_tv_subscriber': 1,
            'is_movie_package_subscriber': 0,
            'subscription_age': 6.0,
            'bill_avg': 80.0,
            'reamining_contract': 2.0,
            'service_failure_count': 2,
            'download_avg': 20.0,
            'upload_avg': 3.0,
            'download_over_limit': 1
        },
        {
            'id': 4,
            'is_tv_subscriber': 0,
            'is_movie_package_subscriber': 1,
            'subscription_age': 4.0,
            'bill_avg': 100.0,
            'reamining_contract': 1.0,
            'service_failure_count': 3,
            'download_avg': 15.0,
            'upload_avg': 2.0,
            'download_over_limit': 1
        },
        # Високий ризик
        {
            'id': 5,
            'is_tv_subscriber': 0,
            'is_movie_package_subscriber': 0,
            'subscription_age': 3.0,
            'bill_avg': 150.0,
            'reamining_contract': 0.1,
            'service_failure_count': 5,
            'download_avg': 5.0,
            'upload_avg': 0.5,
            'download_over_limit': 1
        },
        {
            'id': 6,
            'is_tv_subscriber': 0,
            'is_movie_package_subscriber': 0,
            'subscription_age': 1.0,
            'bill_avg': 200.0,
            'reamining_contract': 0.0,
            'service_failure_count': 8,
            'download_avg': 2.0,
            'upload_avg': 0.1,
            'download_over_limit': 1
        },
        # Екстремальний ризик
        {
            'id': 7,
            'is_tv_subscriber': 0,
            'is_movie_package_subscriber': 0,
            'subscription_age': 0.5,
            'bill_avg': 250.0,
            'reamining_contract': 0.0,
            'service_failure_count': 12,
            'download_avg': 1.0,
            'upload_avg': 0.0,
            'download_over_limit': 1
        }
    ]
    
    # Створення DataFrame
    df = pd.DataFrame(test_data)
    
    # Збереження в CSV
    df.to_csv('test_batch_data.csv', index=False)
    print("Тестовий CSV файл створено: test_batch_data.csv")
    print(f"Кількість клієнтів: {len(df)}")
    print("\nВміст файлу:")
    print(df.to_string())
    
    return df

if __name__ == "__main__":
    create_test_csv()