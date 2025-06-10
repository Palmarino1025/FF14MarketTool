CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            server TEXT NOT NULL
        );
 

INSERT INTO items (id, name, price, timestamp, server) VALUES
    (1, 'Item 1', 10.99, '2023-10-01 12:00:00', 'Server A'),
    (2, 'Item 2', 15.49, '2023-10-02 13:30:00', 'Server B'),
    (3, 'Item 3', 7.25, '2023-10-03 14:45:00', 'Server A'),
    (4, 'Item 4', 20.00, '2023-10-04 16:00:00', 'Server C'),
    (5, 'Item 5', 5.75, '2023-10-05 17:15:00', 'Server B');

