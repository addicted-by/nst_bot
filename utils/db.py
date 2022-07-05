import sqlite3
import os
import os.path

class DBHandler():
    def __init__(self, db_file: str='./users_db.sqlite'):
        super(DBHandler, self).__init__()
        db_is_new = not os.path.exists(db_file)
        self.conn = None
        self.db_file = os.path.abspath(db_file)
        self.connect_db()
        if db_is_new:
            print('Creating schema')
            sql = '''create table if not exists USERS(
            USERID INTEGER PRIMARY KEY,
            CONTENT BLOB,
            STYLE BLOB,
            STYLIZED BLOB,
            STATS BLOB);'''
            self.conn.execute(sql)   
            print('Created')
        else:
            print('Schema exists\n') 
        self.close_db()

    def connect_db(self):
        self.conn = sqlite3.connect(self.db_file)

    def close_db(self):
        self.conn.close()


    def insert_id(self, user_id: int):
        self.connect_db()
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM USERS WHERE USERID = ?", (user_id,))
        data=cursor.fetchall()
        cursor.close()
        if len(data) == 0:
            sql = f"INSERT INTO USERS (USERID, CONTENT, STYLE, STATS, STYLIZED) VALUES(?,?,?,?,?);"
            cursor = self.conn.cursor()
            cursor.execute(sql, [user_id, None, None, None, None])
            self.conn.commit()
            cursor.close()
            print("User added")
        else:
            print("User's existance confirmed")
        self.close_db()

    def insert_picture_by_file(self, user_id: int,
                                    picture_file: str,
                                    picture_type: str) -> int:
        with open(picture_file, 'rb') as input_file:
            ablob = input_file.read()
            self.connect_db()
            sql = f'''UPDATE USERS SET
            {picture_type.upper()} = ? WHERE USERID=?;'''
            data = (sqlite3.Binary(ablob), user_id)
            cursor = self.conn.cursor()
            cursor.execute(sql, data) 
            self.conn.commit()
            print('Updated')
        self.close_db()
    
    def insert_picture_by_bytes(self, user_id: int,
                                      picture_bytes: bytes, 
                                      picture_type: str):
        self.connect_db()
        sql = f'''UPDATE USERS SET
            {picture_type.upper()} = ? WHERE USERID=?;'''
        data = (sqlite3.Binary(picture_bytes), user_id)
        cursor = self.conn.cursor()
        cursor.execute(sql, data) 
        self.conn.commit()
        cursor.close()
        print('Updated')
        self.close_db()

    def insert_stats(self, user_id: int, stats: bytes):
        self.connect_db()
        sql = '''UPDATE USERS SET STATS=? WHERE USERID=?;'''
        data = (sqlite3.Binary(stats), user_id)
        cursor = self.conn.cursor()
        cursor.execute(sql, data)
        self.conn.commit()
        cursor.close()
        print('Stats saved')
        self.close_db()

    def insert_stylized(self, user_id: int, stylized: bytes):
        self.connect_db()
        sql = '''UPDATE USERS SET STYLIZED=? WHERE USERID=?'''
        data = (sqlite3.Binary(stylized), user_id)
        cursor = self.conn.cursor()
        cursor.execute(sql, data)
        self.conn.commit()
        cursor.close()
        print('Stylized saved')
        self.close_db()

    def extract_field(self, user_id: int, field: str):
        self.connect_db()
        sql = 'SELECT {field} FROM USERS WHERE USERID = {value}'.format(
            field=field.upper(),
            value=user_id
        )
        cursor = self.conn.cursor()
        cursor.execute(sql)
        field_value = cursor.fetchone()
        self.close_db()
        return field_value[-1]

    def extract_picture(self, user_id: int,  
                            both: bool=True) -> tuple:
        self.connect_db()
        cursor = self.conn.cursor()
        param = {'id': user_id}
        if both:
            sql = f'SELECT CONTENT, STYLE FROM USERS WHERE USERID = {user_id}'
            cursor.execute(sql, param)
            content, style = cursor.fetchone()
            self.close_db()
            return content, style
        else: 
            sql = f"SELECT CONTENT FROM USERS WHERE USERID = {user_id}"
            cursor.execute(sql, param)
            content = cursor.fetchone()
            self.close_db()
            return content[-1]

    def chk_conn(self) -> bool:
     try:
        self.conn.cursor()
        return True
     except Exception as ex:
        return False

    def __del__(self):
        print('Deleting database')
        if self.chk_conn():
            self.conn.close()
        else:
            print('Deleted')
        # gc.collect()