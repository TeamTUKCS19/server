

def insert_building(conn, cursor, building, side, width, risk, crack_id, date):
    query = "INSERT INTO Building (Building, Side, Width, Risk, Crack_id, Date) VALUES (%s,%s,%s,%s,%s,%s)"
    values = (building, side, width, risk, crack_id, date)
    try:
        cursor.execute(query, values)
        conn.commit()
        print("Building inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting building: {e}")


def insert_location(conn, cursor, crack_id, latitude, longitude, altitude):
    query = "INSERT INTO Location (Crack_id,Latitude,Longitude,Altitude) VALUES(%s,%s,%s,%s)"
    values = (crack_id, latitude, longitude, altitude)
    try:
        cursor.execute(query, values)
        conn.commit()
        print("location inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting location: {e}")


def insert_crackURL(conn, cursor, crack_id, crop_url, bbox_url):
    query = "INSERT INTO Crack_url (Crack_id, Crop_url, Bbox_url) VALUES(%s,%s,%s)"
    values = (crack_id, crop_url, bbox_url)
    try:
        cursor.execute(query, values)
        conn.commit()
        print("Crack_url inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting Crack_url: {e}")


def get_crack_id(cursor, building, side, date):
    try:
        # 1. Crack_id
        crack_id = f'{building}{side}{date}'
        query_crack_id = "SELECT * FROM Building WHERE Crack_id = %s"
        cursor.execute(query_crack_id, crack_id)
        results = cursor.fetchall()

        if not results:
            print("Crack_id not found.")
            return None

        crack_ids = [result[4] for result in results]

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None


# building, side는 문자가아닌 키값이 들어가야함, date에는 Day만 들어가야함. Ex. building = 5, side = 1
# 위를 이용하여 crack_id = f'{building}{side}' 로 주면 됨
def get_location(conn, crack_id):
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM Location WHERE Crack_id LIKE %s"
        cursor.execute(query, (f'{crack_id}%',))
        results = cursor.fetchall()
        if not results:
            print("Crack_id not found.")
            return None
        print("i get results")
        return results

    except Exception as e:
        print(f"Error : {e}")
        return None


def get_crack_url(cursor, crack_id):
    try:
        query = "SELECT * FROM Crack_url WHERE Crack_id LIKE = %s%"
        cursor.execute(query, crack_id)
        results = cursor.fetchall()

        if not results:
            print("Crack_id not found.")
            return None
        return results

    except Exception as e:
        print(f"Error : {e}")
        return None

