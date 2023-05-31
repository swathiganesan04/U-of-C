########################## ASSIGNMENT 4a SQL ##############################

# Name: Swathi Ganesan
# Date: 19-Oct-2022

####### INSTRUCTIONS #######

# Read through the whole template and read each question carefully.  Make sure to follow all instructions.

# Each question should be answered with only one SQL query per question, unless otherwise stated.
# All queries must be written below the corresponding question number.
# Make sure to include the schema name in all table references (i.e. sakila.customer, not just customer)
# DO NOT modify the comment text for each question unless asked.
# Any additional comments you may wish to add to organize your code MUST be on their own lines and each comment line must begin with a # character
# If a question asks for specific columns and/or column aliases, they MUST be followed.
# Pay attention to the requested column aliases for aggregations and calculations. Otherwise, do not re-alias columns from the original column names in the tables unless asked to do so.
# Return columns in the order requested in the question.
# Do not concatenate columns together unless asked.

# Refer to the Sakila documentation for further information about the tables, views, and columns: https://dev.mysql.com/doc/sakila/en/

##########################################################################

## Desc: Joining Data, Nested Queries, Views and Indexes, Transforming Data

############################ PREREQUESITES ###############################

# These queries make use of the Sakila schema.  If you have issues with the Sakila schema, try dropping the schema and re-loading it from the scripts provided with Assignment 2.

# Run the following two SQL statements before beginning the questions:
SET SQL_SAFE_UPDATES=0; 
UPDATE sakila.film SET language_id=6 WHERE title LIKE "%ACADEMY%";

############################### QUESTION 1 ###############################

# 1a) List the actors (first_name, last_name, actor_id) who acted in more then 25 movies.  Also return the count of movies they acted in, aliased as movie_count. Order by first and last name alphabetically.
SELECT 
  b.first_name, 
  b.last_name, 
  a.actor_id, 
  a.movie_count 
FROM 
  (
    SELECT 
      actor_id, 
      COUNT(film_id) as movie_count 
    FROM 
      sakila.film_actor 
    GROUP BY 
      actor_id 
    HAVING 
      movie_count > 25
  ) a 
  LEFT JOIN sakila.actor b ON a.actor_id = b.actor_id 
ORDER BY 
  b.first_name, 
  b.last_name;

# 1b) List the actors (first_name, last_name, actor_id) who have worked in German language movies. Order by first and last name alphabetically.
SELECT 
  d.first_name, 
  d.last_name, 
  c.actor_id 
FROM 
  (
    SELECT 
      actor_id 
    FROM 
      sakila.film_actor 
    WHERE 
      film_id in (
        SELECT 
          a.film_id 
        FROM 
          sakila.film a 
          LEFT JOIN sakila.language b ON a.language_id = b.language_id 
        WHERE 
          b.name = 'GERMAN'
      )
  ) c 
  LEFT JOIN sakila.actor d ON c.actor_id = d.actor_id 
ORDER BY 
  d.first_name, 
  d.last_name;

# 1c) List the actors (first_name, last_name, actor_id) who acted in horror movies and the count of horror movies by each actor.  Alias the count column as horror_movie_count. Order by first and last name alphabetically.
SELECT 
  d.first_name, 
  d.last_name, 
  c.actor_id,
  c.horror_movie_counts
FROM 
  (
    SELECT 
      actor_id, 
      COUNT(film_id) as horror_movie_counts 
    FROM 
      sakila.film_actor 
    WHERE 
      film_id IN (
        SELECT 
          film_id 
        FROM 
          sakila.film_category 
        WHERE 
          category_id IN (
            SELECT 
              category_id 
            FROM 
              sakila.category 
            WHERE 
              name = 'Horror'
          )
      ) 
    GROUP BY 
      actor_id
  ) c 
  LEFT JOIN sakila.actor d ON c.actor_id = d.actor_id 
ORDER BY 
  d.first_name, 
  d.last_name;

# 1d) List the customers who rented more than 3 horror movies.  Return the customer first and last names, customer IDs, and the horror movie rental count (aliased as horror_movie_count). Order by first and last name alphabetically.
SELECT 
  a.first_name, 
  a.last_name, 
  a.customer_id, 
  COUNT(b.rental_id) as horror_movie_count 
FROM 
  customer a 
  LEFT JOIN (
    SELECT 
      customer_id, 
      rental_id 
    FROM 
      sakila.rental 
    WHERE 
      inventory_id IN (
        SELECT 
          inventory_id 
        FROM 
          sakila.inventory 
        WHERE 
          film_id IN (
            SELECT 
              film_id 
            FROM 
              sakila.film_category 
            WHERE 
              category_id IN (
                SELECT 
                  category_id 
                FROM 
                  sakila.category 
                WHERE 
                  name = 'Horror'
              )
          )
      )
  ) b on a.customer_id = b.customer_id 
GROUP BY 
  a.first_name, 
  a.last_name, 
  a.customer_id 
HAVING 
  horror_movie_count > 3 
ORDER BY 
  a.first_name, 
  a.last_name;

# 1e) List the customers who rented a movie which starred Scarlett Bening.  Return the customer first and last names and customer IDs. Order by first and last name alphabetically.
SELECT 
  DISTINCT a.first_name, 
  a.last_name, 
  a.customer_id 
FROM 
  customer a 
  INNER JOIN (
    SELECT 
      DISTINCT customer_id, 
      rental_id 
    FROM 
      sakila.rental 
    WHERE 
      inventory_id IN (
        SELECT 
          inventory_id 
        FROM 
          sakila.inventory 
        WHERE 
          film_id IN (
            SELECT 
              film_id 
            FROM 
              sakila.film_actor 
            WHERE 
              actor_id IN (
                SELECT 
                  actor_id 
                FROM 
                  sakila.actor 
                WHERE 
                  first_name = 'Scarlett' 
                  AND last_name = 'Bening'
              )
          )
      )
  ) b on a.customer_id = b.customer_id 
ORDER BY 
  a.first_name, 
  a.last_name;

# 1f) Which customers residing at postal code 62703 rented movies that were Documentaries?  Return their first and last names and customer IDs.  Order by first and last name alphabetically.
SELECT 
  DISTINCT a.first_name, 
  a.last_name, 
  a.customer_id 
FROM 
  customer a 
  INNER JOIN (
    SELECT 
      DISTINCT customer_id, 
      rental_id 
    FROM 
      sakila.rental 
    WHERE 
      inventory_id IN (
        SELECT 
          inventory_id 
        FROM 
          sakila.inventory 
        WHERE 
          film_id IN (
            SELECT 
              film_id 
            FROM 
              sakila.film_category 
            WHERE 
              category_id IN (
                SELECT 
                  category_id 
                FROM 
                  sakila.category 
                WHERE 
                  name = 'Documentary'
              )
          )
      )
  ) b on a.customer_id = b.customer_id 
WHERE 
  a.customer_id IN (
    SELECT 
      customer_id 
    FROM 
      sakila.customer 
    WHERE 
      address_id IN (
        SELECT 
          address_id 
        FROM 
          sakila.address 
        WHERE 
          postal_code = '62703'
      )
  ) 
ORDER BY 
  a.first_name, 
  a.last_name;

# 1g) Find all the addresses (if any) where the second address line is not empty and is not NULL (i.e., contains some text).  Return the address_id and address_2, sorted by address_2 in ascending order.
SELECT 
  address_id, 
  address2 
FROM 
  sakila.address 
WHERE 
  address2 IS NOT NULL 
  AND address2 != '' 
ORDER BY 
  address2;

# 1h) List the actors (first_name, last_name, actor_id)  who played in a film involving a “Crocodile” and a “Shark” (in the same movie).  Also include the title and release year of the movie.  Sort the results by the actors’ last name then first name, in ascending order.
SELECT 
  d.first_name, 
  d.last_name, 
  c.actor_id, 
  c.title, 
  c.release_year 
FROM 
  (
    SELECT 
      a.*, 
      b.actor_id 
    FROM 
      (
        SELECT 
          film_id, 
          title, 
          release_year 
        FROM 
          sakila.film 
        WHERE 
          description LIKE '%Crocodile%' 
          AND description LIKE '%Shark%'
      ) a 
      INNER JOIN (
        SELECT 
          DISTINCT actor_id, 
          film_id 
        FROM 
          sakila.film_actor
      ) b ON a.film_id = b.film_id
  ) c 
  INNER JOIN sakila.actor d ON c.actor_id = d.actor_id 
ORDER BY 
  d.last_name, 
  d.first_name;

# 1i) Find all the film categories in which there are between 55 and 65 films. Return the category names and the count of films per category, sorted from highest to lowest by the number of films.  Alias the count column as count_movies. Order the results alphabetically by category.
SELECT 
  b.name, 
  a.count_movies 
FROM 
  (
    SELECT 
      category_id, 
      count(film_id) as count_movies 
    FROM 
      sakila.film_category 
    GROUP BY 
      category_id 
    HAVING 
      count(film_id) BETWEEN 55 
      AND 65 
    ORDER BY 
      count(film_id) DESC
  ) a 
  LEFT JOIN sakila.category b on a.category_id = b.category_id 
ORDER BY 
  a.count_movies DESC, 
  b.name;

# 1j) In which of the film categories is the average difference between the film replacement cost and the rental rate larger than $17?  Return the film categories and the average cost difference, aliased as mean_diff_replace_rental.  Order the results alphabetically by category.
SELECT 
  c.name as category, 
  d.mean_diff_replace_rental 
FROM 
  sakila.category c 
  INNER JOIN (
    SELECT 
      DISTINCT a.category_id, 
      AVG(
        b.replacement_cost - b.rental_rate
      ) OVER (PARTITION BY a.category_id) AS mean_diff_replace_rental 
    FROM 
      sakila.film b 
      INNER JOIN sakila.film_category a on a.film_id = b.film_id
  ) d on c.category_id = d.category_id 
WHERE 
  d.mean_diff_replace_rental > 17 
ORDER BY 
  c.name;

# 1k) Create a list of overdue rentals so that customers can be contacted and asked to return their overdue DVDs. Return the title of the film, the customer first name and last name, customer phone number, and the number of days overdue, aliased as days_overdue.  Order the results by first and last name alphabetically.
## NOTE: To identify if a rental is overdue, find rentals that have not been returned and have a rental date rental date further in the past than the film's rental duration (rental duration is in days)
SELECT 
  film.title, 
  customer.first_name, 
  customer.last_name, 
  address.phone, 
  DATEDIFF(
    CURRENT_DATE(), 
    rental.rental_date
  ) - film.rental_duration as days_overdue 
FROM 
  rental 
  INNER JOIN customer ON rental.customer_id = customer.customer_id 
  INNER JOIN address ON customer.address_id = address.address_id 
  INNER JOIN inventory ON rental.inventory_id = inventory.inventory_id 
  INNER JOIN film ON inventory.film_id = film.film_id 
WHERE 
  rental.return_date IS NULL 
  AND DATE_ADD(
    rental_date, INTERVAL film.rental_duration DAY
  ) < CURRENT_DATE() 
ORDER BY 
  customer.first_name, customer.last_name;

# 1l) Find the list of all customers and staff for store_id 1.  Return the first and last names, as well as a column indicating if the name is "staff" or "customer", aliased as person_type.  Order results by first name and last name alphabetically.
## Note : use a set operator and do not remove duplicates
SELECT 
  * 
FROM 
  (
    SELECT DISTINCT
      first_name, 
      last_name, 
      'staff' AS person_type 
    FROM 
      sakila.staff 
    WHERE 
      store_id = 1 
    UNION ALL
    SELECT DISTINCT
      first_name, 
      last_name, 
      'customer' AS person_type 
    FROM 
      sakila.customer 
    WHERE 
      store_id = 1
  ) a 
ORDER BY 
  first_name, 
  last_name;

############################### QUESTION 2 ###############################

# 2a) List the first and last names of both actors and customers whose first names are the same as the first name of the actor with actor_id 8.  Order in alphabetical order by last name.
## NOTE: Do not remove duplicates and do not hard-code the first name in your query.
SELECT 
  * 
FROM 
  (
    SELECT 
      first_name, 
      last_name 
    FROM 
      sakila.actor 
    UNION ALL 
    SELECT 
      first_name, 
      last_name 
    FROM 
      sakila.customer
  ) a 
WHERE 
  first_name = (
    SELECT 
      first_name 
    FROM 
      sakila.actor 
    WHERE 
      actor_id = 8
  ) 
ORDER BY 
  last_name;

# 2b) List customers (first name, last name, customer ID) and payment amounts of customer payments that were greater than average the payment amount.  Sort in descending order by payment amount.
## HINT: Use a subquery to help
SELECT 
  a.first_name, 
  a.last_name, 
  a.customer_id, 
  b.amount 
FROM 
  sakila.customer a 
  INNER JOIN sakila.payment b ON a.customer_id = b.customer_id 
WHERE 
  b.amount > (
    SELECT 
      AVG(amount) 
    FROM 
      sakila.payment
  ) 
ORDER BY 
  b.amount DESC;

#### Note : The solutions contains customer payments that are greater than average. Since some customers have multiple rentals, the result returns 7746 rows with duplicate customer entries

# 2c) List customers (first name, last name, customer ID) who have rented movies at least once.  Order results by first name, lastname alphabetically.
## Note: use an IN clause with a subquery to filter customers
SELECT 
  DISTINCT first_name, 
  last_name, 
  customer_id 
FROM 
  sakila.customer 
WHERE 
  customer_id IN (
    SELECT 
      DISTINCT customer_id 
    FROM 
      sakila.rental
  ) 
ORDER BY 
  first_name, 
  last_name;

# 2d) Find the floor of the maximum, minimum and average payment amount.  Alias the result columns as max_payment, min_payment, avg_payment.
SELECT 
  FLOOR(
    MAX(amount)
  ) AS max_payment, 
  FLOOR(
    MIN(amount)
  ) AS min_payment, 
  FLOOR(
    AVG(amount)
  ) AS avg_payment 
FROM 
  sakila.payment;

############################### QUESTION 3 ###############################

# 3a) Create a view called actors_portfolio which contains the following columns of information about actors and their films: actor_id, first_name, last_name, film_id, title, category_id, category_name
CREATE VIEW actors_portfolio AS (
  SELECT 
    h.actor_id, 
    g.first_name, 
    g.last_name, 
    h.film_id, 
    h.title, 
    h.category_id, 
    h.category_name 
  FROM 
    (
      SELECT 
        e.actor_id, 
        f.* 
      FROM 
        sakila.film_actor e 
        LEFT JOIN (
          SELECT 
            c.film_id, 
            c.title, 
            c.category_id, 
            d.name as category_name 
          FROM 
            (
              SELECT 
                a.film_id, 
                a.title, 
                b.category_id 
              FROM 
                sakila.film a 
                LEFT JOIN sakila.film_category b ON a.film_id = b.film_id
            ) c 
            LEFT JOIN sakila.category d ON c.category_id = d.category_id
        ) f ON e.film_id = f.film_id
    ) h 
    LEFT JOIN sakila.actor g on h.actor_id = g.actor_id
);

# 3b) Describe (using a SQL command) the structure of the view.
DESC actors_portfolio;

# 3c) Query the view to get information (all columns) on the actor ADAM GRANT
SELECT 
  * 
FROM 
  actors_portfolio 
WHERE 
  first_name = 'ADAM' 
  AND last_name = 'GRANT';

# 3d) Insert a new movie titled Data Hero in Sci-Fi Category starring ADAM GRANT
## NOTE: If you need to use multiple statements for this question, you may do so.
## WARNING: Do not hard-code any id numbers in your where criteria.
## !! Think about how you might do this before reading the hints below !!
## HINT 1: Given what you know about a view, can you insert directly into the view? Or do you need to insert the data elsewhere?
## HINT 2: Consider using SET and LAST_INSERT_ID() to set a variable to aid in your process.

INSERT INTO sakila.film(title, language_id) 
VALUES 
  (
    'Data Hero', 
    (
      SELECT 
        language_id 
      FROM 
        sakila.language 
      WHERE 
        name = 'English'
    )
  );

SET @last_insert_film_id = last_insert_id();

INSERT INTO sakila.film_actor(actor_id, film_id) 
VALUES 
  (
    (
      SELECT 
        actor_id 
      FROM 
        sakila.actor 
      where 
        first_name = 'ADAM' 
        AND last_name = 'GRANT'
    ), @last_insert_film_id
  );

INSERT INTO sakila.film_category(film_id, category_id) 
VALUES 
  (
    (
      SELECT 
        film_id 
      FROM 
        sakila.film 
      WHERE 
        title = 'Data Hero'
    ), 
    (
      SELECT 
        category_id 
      FROM 
        sakila.category 
      WHERE 
        name = 'Sci-Fi'
    )
  );

SELECT 
  * 
FROM 
  actors_portfolio 
WHERE 
  first_name = 'ADAM' 
  AND last_name = 'GRANT';
  
############################### QUESTION 4 ###############################

# 4a) Extract the street number (numbers at the beginning of the address) from the customer address in the customer_list view.  Return the original address column, and the street number column (aliased as street_number).  Order the results in ascending order by street number.
## NOTE: Use Regex to parse the street number
SELECT 
  address, 
  CAST(
    REGEXP_SUBSTR(address, '[0-9]+') AS UNSIGNED
  ) as street_number 
FROM 
  sakila.customer_list 
ORDER BY 
  street_number;

# 4b) List actors (first name, last name, actor id) whose last name starts with characters A, B or C.  Order by first_name, last_name in ascending order.
## NOTE: Use either a LEFT() or RIGHT() operator
SELECT 
  first_name, 
  last_name, 
  actor_id 
FROM 
  sakila.actor 
WHERE 
  LEFT(last_name, 1) IN ('A', 'B', 'C') 
ORDER BY 
  first_name, 
  last_name;

# 4c) List film titles that contain exactly 10 characters.  Order titles in ascending alphabetical order.
SELECT 
  title 
FROM 
  sakila.film 
WHERE 
  LENGTH(title) = 10 
ORDER BY 
  title;

# 4d) Return a list of distinct payment dates formatted in the date pattern that matches "22/01/2016" (2 digit day, 2 digit month, 4 digit year).  Alias the formatted column as payment_date.  Return the formatted dates in ascending order.
SELECT 
  payment_date 
FROM 
  (
    SELECT 
      DISTINCT DATE_FORMAT(payment_date, '%e/%m/%Y') as payment_date, 
      SUBSTR(payment_date, 1, 10) as pmt_order 
    FROM 
      sakila.payment 
    ORDER BY 
      pmt_order
  ) a;

# 4e) Find the number of days each rental was out (days between rental_date & return_date), for all returned rentals.  Return the rental_id, rental_date, return_date, and alias the days between column as days_out.  Order with the longest number of days_out first.
SELECT 
  rental_id, 
  rental_date, 
  return_date, 
  DATEDIFF(return_date, rental_date) AS days_out 
FROM 
  sakila.rental 
WHERE 
  return_date IS NOT NULL 
ORDER BY 
  days_out DESC;
  