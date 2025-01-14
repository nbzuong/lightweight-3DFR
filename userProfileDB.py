import sqlite3
import numpy as np
import datetime
import os
import pandas as pd
import torch
import io

# Serialize the tensor
def serialize_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return buffer.read()

# Deserialize the tensor
def deserialize_tensor(blob):
    buffer = io.BytesIO(blob)
    buffer.seek(0)
    return torch.load(buffer)

class UserProfileDB:
    def __init__(self, db_folder="database"):
        self.db_folder = os.path.join(os.path.dirname(__file__), db_folder)
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)
        self.db_file = os.path.join(self.db_folder, "face_recognition.db")
        self.conn = sqlite3.connect(self.db_file)
        self.create_tables()

    def create_tables(self):
        """Creates the necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                dob TEXT,
                phone_number TEXT,
                registration_date TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                user_id INTEGER,
                embedding2d BLOB,
                embedding3d BLOB,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """
        )
        self.conn.commit()

    def user_exists(self, phone_number):
        """
        Checks if a user exists in the database based on their phone number.

        Args:
            phone_number (str): The phone number to check.

        Returns:
            bool: True if a user with the given phone number exists, False otherwise.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT 1 FROM users WHERE phone_number=?",
                (phone_number,),
            )
            result = cursor.fetchone()
            return result is not None  # User exists if a row is returned
        except Exception as e:
            print(f"Error checking user existence: {e}")
            return False
    
    def add_user(self, name, dob, phone_number, face_embedding2d, face_embedding3d):
        """
        Adds a new user to the database.

        Args:
            name (str): The user's name.
            dob (str): The user's date of birth (e.g., "YYYY-MM-DD").
            phone_number (str): The user's phone number.
            face_embedding2d (numpy.ndarray): The 2D face embedding.
            face_embedding3d (numpy.ndarray): The 3D face embedding.
        """
        face_embedding2d = serialize_tensor(face_embedding2d)
        face_embedding3d = serialize_tensor(face_embedding3d)
        
        if self.user_exists(phone_number):
            print(f"User with phone number {phone_number} already exists.")
            return None
        
        registration_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (name, dob, phone_number, registration_date) VALUES (?, ?, ?, ?)",
                (name, dob, phone_number, registration_date),
            )
            user_id = cursor.lastrowid  # Get the auto-generated user_id

            cursor.execute(
                "INSERT INTO embeddings (user_id, embedding2d, embedding3d) VALUES (?, ?, ?)",
                (user_id, face_embedding2d, face_embedding3d),
            )

            self.conn.commit()
            print(f"User {name} (ID: {user_id}) added successfully.")
            return user_id
        except Exception as e:
            print(f"Error adding user: {e}")
            self.conn.rollback()
            return None

    def update_user_data(self, name, dob, phone_number, updated_data):
      """
      Updates user information based on name, dob, and phone number.

      Args:
          name (str): The user's name.
          dob (str): The user's date of birth.
          phone_number (str): The user's phone number.
          updated_data (dict): A dictionary containing the fields to update (e.g., {"name": "New Name", "phone_number": "New Phone"}).

      Returns:
          bool: True if the user was found and updated, False otherwise.
      """
      cursor = self.conn.cursor()
      try:
          cursor.execute(
              "SELECT user_id FROM users WHERE name=? AND dob=? AND phone_number=?",
              (name, dob, phone_number),
          )
          result = cursor.fetchone()
          if result:
              user_id = result[0]
              update_fields = []
              update_values = []

              # Update user information fields if provided
              for key, value in updated_data.items():
                  if key in ["name", "dob", "phone_number"]:
                      update_fields.append(f"{key}=?")
                      update_values.append(value)

              if update_fields:
                  update_query = f"UPDATE users SET {', '.join(update_fields)} WHERE user_id=?"
                  update_values.append(user_id)
                  cursor.execute(update_query, tuple(update_values))

              self.conn.commit()
              print(f"User data for {name} (ID: {user_id}) updated successfully.")
              return True
          else:
              print(f"User with name: {name}, DOB: {dob}, and phone: {phone_number} not found.")
              return False

      except Exception as e:
          print(f"Error updating user data: {e}")
          self.conn.rollback()
          return False

    def save_database_backup(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_db_file = os.path.join(self.db_folder, f"face_recognition_backup_{current_time}.db")

        # Create a new connection to copy the database
        backup_conn = sqlite3.connect(backup_db_file)
        with backup_conn:
            self.conn.backup(backup_conn)
        backup_conn.close()

        print(f"Database saved as {backup_db_file}")

        # Keep only the 3 most recent backup files
        self.cleanup_old_backups()

    def cleanup_old_backups(self):
        backup_files = [
            f for f in os.listdir(self.db_folder) if f.startswith("face_recognition_backup_") and f.endswith(".db")
        ]
        backup_files.sort(reverse=True)  # Sort by modification time (newest first)

        while len(backup_files) > 3:
            oldest_backup = os.path.join(self.db_folder, backup_files.pop())
            os.remove(oldest_backup)
            print(f"Deleted old backup: {oldest_backup}")

    def get_embeddings_dataframe(self):
        """
        Retrieves user_id, embedding2d, and embedding3d for all users from the database
        and returns them as a Pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing user_id, embedding2d, and embedding3d,
                              or None if an error occurs.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT user_id, embedding2d, embedding3d FROM embeddings"
            )
            results = cursor.fetchall()

            if not results:
                print("No embeddings found in the database.")
                return pd.DataFrame(columns=["user_id", "embedding2d", "embedding3d"])

            # Convert BLOBs back to numpy arrays
            data = []
            for row in results:
                user_id, embedding2d_blob, embedding3d_blob = row
                embedding2d = deserialize_tensor(embedding2d_blob)
                embedding3d = deserialize_tensor(embedding3d_blob)
                
                data.append([user_id, embedding2d, embedding3d])
            
            df = pd.DataFrame(data, columns=["user_id", "embedding2d", "embedding3d"])
            return df
        
        except Exception as e:
            print(f"Error retrieving embeddings: {e}")
            return None

    def get_user_info(self, user_id):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT name, dob, phone_number, registration_date FROM users WHERE user_id=?",
                (user_id,)
            )
            result = cursor.fetchone()

            if result:
                user_info = {
                    "name": result[0],
                    "dob": result[1],
                    "phone_number": result[2],
                    "registration_date": result[3]
                }
                return user_info
            else:
                print(f"User with ID {user_id} not found.")
                return None

        except Exception as e:
            print(f"Error retrieving user information: {e}")
            return None
        
        
    def close_connection(self):
        """
        Close the connection to the database
        """
        self.conn.close()
        print("Database connection closed.")
