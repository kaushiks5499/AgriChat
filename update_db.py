"""
Database Update Module for Cotton Trial Data
Handles Excel file processing and Supabase database updates
"""

import pandas as pd
import streamlit as st
from supabase import create_client
from typing import Tuple, Optional


def get_supabase_credentials() -> Tuple[str, str]:
    """
    Get Supabase credentials from Streamlit secrets or environment variables
    
    Returns:
        Tuple[str, str]: (SUPABASE_URL, SUPABASE_KEY)
    """
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    except:
        import os
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    
    return SUPABASE_URL, SUPABASE_KEY


def validate_filename(filename: str) -> Tuple[bool, Optional[float], str]:
    """
    Validate that filename is in '[year] .xlsx' format
    
    Args:
        filename: The uploaded file name
        
    Returns:
        Tuple[bool, Optional[float], str]: (is_valid, year, error_message)
    """
    if not filename.endswith('.xlsx'):
        return False, None, "File must be an Excel file (.xlsx)"
    
    try:
        # Extract year from filename (e.g., "2024 County RACE.xlsx" -> "2024")
        year_str = filename.split()[0]
        year = float(year_str)
        
        # Basic validation - year should be reasonable
        if year < 2000 or year > 2100:
            return False, None, f"Year {year} seems invalid. Please use format: '[YEAR] description.xlsx'"
        
        return True, year, ""
    except (IndexError, ValueError):
        return False, None, "Filename must start with a year (e.g., '2024 County RACE.xlsx')"


def process_excel_file(file_content, filename: str) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """
    Process uploaded Excel file and prepare data for insertion
    
    Args:
        file_content: The uploaded file content (BytesIO object)
        filename: The name of the uploaded file
        
    Returns:
        Tuple[bool, Optional[pd.DataFrame], str]: (success, dataframe, message)
    """
    try:
        # Validate filename format
        is_valid, year, error_msg = validate_filename(filename)
        if not is_valid:
            return False, None, error_msg
        
        # Read all sheets from Excel file
        excel = pd.read_excel(file_content, sheet_name=None)
        
        if not excel:
            return False, None, "No sheets found in Excel file"
        
        # Process each sheet
        to_be_inserted = pd.DataFrame()
        
        for sheet_name in excel.keys():
            try:
                sheet_df = excel[sheet_name]
                
                # Check if required columns exist
                required_cols = ['Variety', 'Yield', 'Turnout', 'Mic', 'Length', 
                               'Strength', 'Uniformity', 'Loan ', 'Lint value']
                missing_cols = [col for col in required_cols if col not in sheet_df.columns]
                
                if missing_cols:
                    return False, None, f"Sheet '{sheet_name}' is missing columns: {missing_cols}"
                
                # Group by variety and calculate means
                temp = sheet_df.groupby('Variety', as_index=True).mean().sort_values('Yield', ascending=False)[
                    ['Yield', 'Turnout', 'Mic', 'Length', 'Strength', 'Uniformity', 'Loan ', 'Lint value']
                ]
                
                # Add trial location and year
                temp['TrialLocation'] = sheet_name
                temp['Year'] = year
                
                # Concatenate to main dataframe
                to_be_inserted = pd.concat([to_be_inserted, temp])
                
            except Exception as e:
                return False, None, f"Error processing sheet '{sheet_name}': {str(e)}"
        
        # Reset index and rename columns to match database schema
        to_be_inserted = to_be_inserted.reset_index()
        to_be_inserted.columns = [
            'Variety', 'Lint', 'Turnout', 'Micronaire', 'Length', 'Strength',
            'Uniformity', 'LoanValue', 'LintValue', 'TrialLocation', 'Year'
        ]
        
        return True, to_be_inserted, f"Successfully processed {len(to_be_inserted)} records from {len(excel)} sheets"
        
    except Exception as e:
        return False, None, f"Error processing Excel file: {str(e)}"


def insert_to_supabase(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Insert processed data into Supabase database
    
    Args:
        df: DataFrame to insert (without 'id' column)
        
    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        # Get Supabase credentials
        SUPABASE_URL, SUPABASE_KEY = get_supabase_credentials()
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            return False, "Supabase credentials not found in secrets or environment variables"
        
        # Create Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get current data to determine last ID
        response = supabase.table("VARIETY_YIELD").select("id").order("id", desc=True).limit(1).execute()
        
        if response.data:
            last_id = int(response.data[0]['id'])
        else:
            last_id = 0
        
        # Add ID column
        df_with_id = df.copy()
        df_with_id['id'] = range(last_id + 1, last_id + len(df) + 1)
        
        # Reorder columns to have 'id' first
        columns = ['id'] + [col for col in df.columns]
        df_with_id = df_with_id[columns]
        
        # Convert DataFrame to list of dictionaries for Supabase insert
        records = df_with_id.to_dict('records')
        
        # Insert data in batches (Supabase has a limit on batch size)
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            response = supabase.table("VARIETY_YIELD").insert(batch).execute()
            total_inserted += len(batch)
        
        return True, f"Successfully inserted {total_inserted} records into database"
        
    except Exception as e:
        return False, f"Error inserting data to Supabase: {str(e)}"


def update_database_from_excel(file_content, filename: str) -> Tuple[bool, str]:
    """
    Main function to process Excel file and update database
    
    Args:
        file_content: The uploaded file content (BytesIO object)
        filename: The name of the uploaded file
        
    Returns:
        Tuple[bool, str]: (success, message)
    """
    # Step 1: Process Excel file
    success, df, message = process_excel_file(file_content, filename)
    
    if not success:
        return False, message
    
    # Step 2: Insert to Supabase
    success, insert_message = insert_to_supabase(df)
    
    if not success:
        return False, insert_message
    
    # Return combined success message
    combined_message = f"{message}\n{insert_message}"
    return True, combined_message