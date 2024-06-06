from torch_frame import stype

task_to_stypes = {
    'rel-stack-user-engagement': {
        'OwnerUserId': stype.numerical,
        'timestamp': stype.timestamp,
        'contribution': stype.categorical,
        'months_since_account_creation': stype.numerical,
        'display_name_is_null': stype.categorical,
        'website_url_is_null': stype.categorical,
        'about_me_length': stype.numerical,
        'location_is_null': stype.categorical,
        'num_badges': stype.numerical,
        'badge_score': stype.numerical,
        'weeks_since_last_comment': stype.numerical,
        'num_comments': stype.numerical,
        'num_posts_commented': stype.numerical,
        'avg_comment_length': stype.numerical,
        'last_q_weeks_ago': stype.numerical,
        'last_q_has_accepted_ans': stype.categorical,
        'last_q_num_tags': stype.numerical,
        'last_q_title_length': stype.numerical,
        'last_q_body_length': stype.numerical,
        'last_q_num_positive_votes': stype.numerical,
        'last_q_num_negative_votes': stype.numerical,
        'last_q_num_comments': stype.numerical,
        'last_q_avg_comment_length': stype.numerical,
        'last_q_num_distinct_commenters': stype.numerical,
        'last_q_avg_commenter_badge_score': stype.numerical,
        'num_questions_last_6mo': stype.numerical,
        'avg_has_accepted_ans': stype.numerical,
        'avg_days_since_last_post_q': stype.numerical,
        'avg_num_tags': stype.numerical,
        'avg_title_length_q': stype.numerical,
        'avg_body_length_q': stype.numerical,
        'avg_num_positive_votes_q': stype.numerical,
        'avg_num_negative_votes_q': stype.numerical,
        'avg_num_comments_q': stype.numerical,
        'avg_avg_comment_length_q': stype.numerical,
        'avg_num_distinct_commenters_q': stype.numerical,
        'avg_commenter_badge_score_q': stype.numerical,
        'last_a_weeks_ago': stype.numerical,
        'last_a_is_accepted_ans': stype.categorical,
        'last_a_body_length': stype.numerical,
        'last_a_num_positive_votes': stype.numerical,
        'last_a_num_negative_votes': stype.numerical,
        'last_a_num_comments': stype.numerical,
        'last_a_avg_comment_length': stype.numerical,
        'last_a_num_distinct_commenters': stype.numerical,
        'last_a_avg_commenter_badge_score': stype.numerical,
        'num_answers_last_6mo': stype.numerical,
        'ans_acceptance_rate': stype.numerical,
        'avg_days_since_last_post_a': stype.numerical,
        'avg_body_length_a': stype.numerical,
        'avg_num_positive_votes_a': stype.numerical,
        'avg_num_negative_votes_a': stype.numerical,
        'avg_num_comments_a': stype.numerical,
        'avg_avg_comment_length_a': stype.numerical,
        'avg_num_distinct_commenters_a': stype.numerical,
        'avg_commenter_badge_score_a': stype.numerical
    },

    'rel-stack-user-badge': {
        'UserId': stype.numerical,
        'timestamp': stype.timestamp,
        'WillGetBadge': stype.categorical,
        'months_since_account_creation': stype.numerical,
        'num_badges': stype.numerical,
        'badge_score': stype.numerical,
        'max_rarity': stype.numerical,
        'avg_rarity': stype.numerical,
        'last_badge_name': stype.categorical,
        'rarest_badge_age_weeks': stype.numerical,
        'last_badge_rarity': stype.numerical,
        'last_badge_weeks_ago': stype.numerical,
        'avg_badge_age_weeks': stype.numerical,
        'avg_weeks_bw_badges': stype.numerical,
        'badge_momentum': stype.numerical,
        'weeks_since_last_comment': stype.numerical,
        'num_comments': stype.numerical,
        'num_posts_commented': stype.numerical,
        'avg_comment_length': stype.numerical,
        'last_q_weeks_ago': stype.numerical,
        'last_q_num_tags': stype.numerical,
        'last_q_body_length': stype.numerical,
        'last_q_num_positive_votes': stype.numerical,
        'last_q_num_negative_votes': stype.numerical,
        'last_q_num_comments': stype.numerical,
        'last_q_avg_commenter_badge_score': stype.numerical,
        'num_questions_last_yr': stype.numerical,
        'avg_days_since_last_post_q': stype.numerical,
        'avg_num_tags': stype.numerical,
        'avg_body_length_q': stype.numerical,
        'avg_num_positive_votes_q': stype.numerical,
        'avg_num_negative_votes_q': stype.numerical,
        'avg_num_comments_q': stype.numerical,
        'avg_commenter_badge_score_q': stype.numerical,
        'last_a_weeks_ago': stype.numerical,
        'last_a_body_length': stype.numerical,
        'last_a_num_positive_votes': stype.numerical,
        'last_a_num_negative_votes': stype.numerical,
        'last_a_num_comments': stype.numerical,
        'last_a_avg_commenter_badge_score': stype.numerical,
        'num_answers_last_yr': stype.numerical,
        'avg_days_since_last_post_a': stype.numerical,
        'avg_body_length_a': stype.numerical,
        'avg_num_positive_votes_a': stype.numerical,
        'avg_num_negative_votes_a': stype.numerical,
        'avg_num_comments_a': stype.numerical,
        'avg_commenter_badge_score_a': stype.numerical
    },

    'rel-stack-post-votes': {
        'PostId': stype.numerical,
        'timestamp': stype.timestamp,
        'popularity': stype.numerical,
        'post_type': stype.categorical,
        'post_age_weeks': stype.numerical,
        'title_length': stype.numerical,
        'body_length': stype.numerical,
        'num_tags': stype.numerical,
        'user_age_months': stype.numerical,
        'post_ordinal': stype.numerical,
        'num_votes': stype.numerical,
        'closed_weeks_ago': stype.numerical,
        'reopened_weeks_ago': stype.numerical,
        'deleted_weeks_ago': stype.numerical,
        'undeleted_weeks_ago': stype.numerical,
        'locked_weeks_ago': stype.numerical,
        'unlocked_weeks_ago': stype.numerical,
        'tweeted_weeks_ago': stype.numerical,
        'bumped_weeks_ago': stype.numerical,
        'avg_owner_question_upvotes_first_month': stype.numerical,
        'avg_owner_question_comments_first_month': stype.numerical,
        'avg_owner_answer_upvotes_first_month': stype.numerical,
        'avg_owner_answer_comments_first_month': stype.numerical
    },

    'rel-amazon-churn': {
        'customer_id': stype.numerical,
        'timestamp': stype.timestamp,
        'churn': stype.categorical,
        'num_reviews': stype.numerical,
        'sum_review_ratings': stype.numerical,
        'avg_review_length': stype.numerical,
        'last_review_weeks_ago': stype.numerical,
        'last_review_summary_text': stype.text_embedded,
        'last_reviewed_product_title': stype.text_embedded,
        'last_reviewed_product_category': stype.categorical,
        'last_review_is_verified': stype.categorical,
        'avg_review_rating': stype.numerical,
        'pct_verified_reviews': stype.numerical,
        'std_review_rating': stype.numerical,
        'min_review_rating': stype.numerical,
        'max_review_rating': stype.numerical,
        'avg_reviewed_product_rating': stype.numerical,
        'sum_reviewed_product_rating': stype.numerical,
        'std_reviewed_product_rating': stype.numerical,
        'min_reviewed_product_rating': stype.numerical,
        'max_reviewed_product_rating': stype.numerical,
        'avg_reviewed_product_price': stype.numerical,
        'sum_reviewed_product_price': stype.numerical,
        'std_reviewed_product_price': stype.numerical,
        'min_reviewed_product_price': stype.numerical,
        'max_reviewed_product_price': stype.numerical,
        'reviewed_product_modal_category': stype.categorical,
        'user_bias': stype.numerical,
        'num_reviews_trend': stype.numerical,
        'avg_rating_trend': stype.numerical,
        'avg_price_trend': stype.numerical,
        'avg_user_bias_trend': stype.numerical
    },

    'rel-amazon-product-churn': {
        'product_id': stype.numerical,
        'timestamp': stype.timestamp,
        'churn': stype.categorical,
        'price': stype.numerical,
        'category': stype.categorical,
        'title': stype.text_embedded,
        'weeks_since_first_review': stype.numerical,
        'weeks_since_median_review': stype.numerical,
        'weeks_since_latest_review': stype.numerical,
        'num_reviews': stype.numerical,
        'sum_ratings': stype.numerical,
        'avg_rating': stype.numerical,
        'std_rating': stype.numerical,
        'min_rating': stype.numerical,
        'max_rating': stype.numerical,
        'pct_verified_reviews': stype.numerical,
        'avg_review_length': stype.numerical,
        'last_review_summary': stype.text_embedded,
        'product_bias': stype.numerical,
        'avg_reviewer_num_reviews': stype.numerical,
        'avg_reviewer_total_spent': stype.numerical,
        'avg_reviewer_avg_price': stype.numerical,
        'avg_reviewer_avg_rating': stype.numerical,
        'avg_reviewer_std_rating': stype.numerical,
        'num_reviews_trend': stype.numerical,
        'avg_rating_trend': stype.numerical,
        'sum_ratings_trend': stype.numerical,
        'min_rating_trend': stype.numerical,
        'max_rating_trend': stype.numerical,
        'avg_review_length_trend': stype.numerical,
        'product_bias_trend': stype.numerical
    },

    'rel-amazon-ltv': {
        'customer_id': stype.numerical,
        'timestamp': stype.timestamp,
        'ltv': stype.numerical,
        'num_reviews_0_to_3': stype.numerical,
        'sum_review_ratings_0_to_3': stype.numerical,
        'avg_review_length_0_to_3': stype.numerical,
        'avg_review_rating_0_to_3': stype.numerical,
        'pct_verified_reviews_0_to_3': stype.numerical,
        'std_review_rating_0_to_3': stype.numerical,
        'min_review_rating_0_to_3': stype.numerical,
        'max_review_rating_0_to_3': stype.numerical,
        'avg_reviewed_product_rating_0_to_3': stype.numerical,
        'sum_reviewed_product_rating_0_to_3': stype.numerical,
        'std_reviewed_product_rating_0_to_3': stype.numerical,
        'min_reviewed_product_rating_0_to_3': stype.numerical,
        'max_reviewed_product_rating_0_to_3': stype.numerical,
        'avg_reviewed_product_price_0_to_3': stype.numerical,
        'sum_reviewed_product_price_0_to_3': stype.numerical,
        'std_reviewed_product_price_0_to_3': stype.numerical,
        'min_reviewed_product_price_0_to_3': stype.numerical,
        'max_reviewed_product_price_0_to_3': stype.numerical,
        'reviewed_product_modal_category_0_to_3': stype.categorical,
        'num_reviews_3_to_6': stype.numerical,
        'sum_review_ratings_3_to_6': stype.numerical,
        'avg_review_length_3_to_6': stype.numerical,
        'avg_review_rating_3_to_6': stype.numerical,
        'pct_verified_reviews_3_to_6': stype.numerical,
        'std_review_rating_3_to_6': stype.numerical,
        'min_review_rating_3_to_6': stype.numerical,
        'max_review_rating_3_to_6': stype.numerical,
        'avg_reviewed_product_rating_3_to_6': stype.numerical,
        'sum_reviewed_product_rating_3_to_6': stype.numerical,
        'std_reviewed_product_rating_3_to_6': stype.numerical,
        'min_reviewed_product_rating_3_to_6': stype.numerical,
        'max_reviewed_product_rating_3_to_6': stype.numerical,
        'avg_reviewed_product_price_3_to_6': stype.numerical,
        'sum_reviewed_product_price_3_to_6': stype.numerical,
        'std_reviewed_product_price_3_to_6': stype.numerical,
        'min_reviewed_product_price_3_to_6': stype.numerical,
        'max_reviewed_product_price_3_to_6': stype.numerical,
        'reviewed_product_modal_category_3_to_6': stype.categorical,
        'num_reviews_6_to_9': stype.numerical,
        'sum_review_ratings_6_to_9': stype.numerical,
        'avg_review_length_6_to_9': stype.numerical,
        'avg_review_rating_6_to_9': stype.numerical,
        'pct_verified_reviews_6_to_9': stype.numerical,
        'std_review_rating_6_to_9': stype.numerical,
        'min_review_rating_6_to_9': stype.numerical,
        'max_review_rating_6_to_9': stype.numerical,
        'avg_reviewed_product_rating_6_to_9': stype.numerical,
        'sum_reviewed_product_rating_6_to_9': stype.numerical,
        'std_reviewed_product_rating_6_to_9': stype.numerical,
        'min_reviewed_product_rating_6_to_9': stype.numerical,
        'max_reviewed_product_rating_6_to_9': stype.numerical,
        'avg_reviewed_product_price_6_to_9': stype.numerical,
        'sum_reviewed_product_price_6_to_9': stype.numerical,
        'std_reviewed_product_price_6_to_9': stype.numerical,
        'min_reviewed_product_price_6_to_9': stype.numerical,
        'max_reviewed_product_price_6_to_9': stype.numerical,
        'reviewed_product_modal_category_6_to_9': stype.categorical,
        'weeks_since_first_review': stype.numerical,
        'last_review_weeks_ago': stype.numerical,
        'last_review_summary_text': stype.text_embedded,
        'last_reviewed_product_title': stype.text_embedded,
        'last_reviewed_product_category': stype.categorical,
        'last_review_is_verified': stype.categorical,
        'num_reviews': stype.numerical,
        'sum_review_ratings': stype.numerical,
        'avg_review_length': stype.numerical,
        'avg_review_rating': stype.numerical,
        'pct_verified_reviews': stype.numerical,
        'std_review_rating': stype.numerical,
        'min_review_rating': stype.numerical,
        'max_review_rating': stype.numerical,
        'avg_reviewed_product_rating': stype.numerical,
        'sum_reviewed_product_rating': stype.numerical,
        'std_reviewed_product_rating': stype.numerical,
        'min_reviewed_product_rating': stype.numerical,
        'max_reviewed_product_rating': stype.numerical,
        'avg_reviewed_product_price': stype.numerical,
        'sum_reviewed_product_price': stype.numerical,
        'std_reviewed_product_price': stype.numerical,
        'min_reviewed_product_price': stype.numerical,
        'max_reviewed_product_price': stype.numerical,
        'reviewed_product_modal_category': stype.categorical
    },

    'rel-amazon-product-ltv': {
        'product_id': stype.numerical,
        'timestamp': stype.timestamp,
        'price': stype.numerical,
        'category': stype.categorical,
        'title': stype.text_embedded,
        'ltv': stype.numerical,
        'num_reviews_0_to_3': stype.numerical,
        'sum_ratings_0_to_3': stype.numerical,
        'avg_rating_0_to_3': stype.numerical,
        'std_rating_0_to_3': stype.numerical,
        'min_rating_0_to_3': stype.numerical,
        'max_rating_0_to_3': stype.numerical,
        'avg_review_length_0_to_3': stype.numerical,
        'pct_verified_reviews_0_to_3': stype.numerical,
        'product_bias_0_to_3': stype.numerical,
        'avg_reviewer_num_reviews_0_to_3': stype.numerical,
        'avg_reviewer_total_spent_0_to_3': stype.numerical,
        'avg_reviewer_avg_price_0_to_3': stype.numerical,
        'avg_reviewer_avg_rating_0_to_3': stype.numerical,
        'avg_reviewer_std_rating_0_to_3': stype.numerical,
        'num_reviews_3_to_6': stype.numerical,
        'sum_ratings_3_to_6': stype.numerical,
        'avg_rating_3_to_6': stype.numerical,
        'std_rating_3_to_6': stype.numerical,
        'min_rating_3_to_6': stype.numerical,
        'max_rating_3_to_6': stype.numerical,
        'avg_review_length_3_to_6': stype.numerical,
        'pct_verified_reviews_3_to_6': stype.numerical,
        'product_bias_3_to_6': stype.numerical,
        'avg_reviewer_num_reviews_3_to_6': stype.numerical,
        'avg_reviewer_total_spent_3_to_6': stype.numerical,
        'avg_reviewer_avg_price_3_to_6': stype.numerical,
        'avg_reviewer_avg_rating_3_to_6': stype.numerical,
        'avg_reviewer_std_rating_3_to_6': stype.numerical,
        'num_reviews_6_to_9': stype.numerical,
        'sum_ratings_6_to_9': stype.numerical,
        'avg_rating_6_to_9': stype.numerical,
        'std_rating_6_to_9': stype.numerical,
        'min_rating_6_to_9': stype.numerical,
        'max_rating_6_to_9': stype.numerical,
        'avg_review_length_6_to_9': stype.numerical,
        'pct_verified_reviews_6_to_9': stype.numerical,
        'product_bias_6_to_9': stype.numerical,
        'avg_reviewer_num_reviews_6_to_9': stype.numerical,
        'avg_reviewer_total_spent_6_to_9': stype.numerical,
        'avg_reviewer_avg_price_6_to_9': stype.numerical,
        'avg_reviewer_avg_rating_6_to_9': stype.numerical,
        'avg_reviewer_std_rating_6_to_9': stype.numerical,
        'weeks_since_first_review': stype.numerical,
        'weeks_since_median_review': stype.numerical,
        'weeks_since_latest_review': stype.numerical,
        'last_review_summary': stype.text_embedded,
        'last_review_is_verified': stype.categorical,
        'num_reviews': stype.numerical,
        'sum_ratings': stype.numerical,
        'avg_rating': stype.numerical,
        'std_rating': stype.numerical,
        'min_rating': stype.numerical,
        'max_rating': stype.numerical,
        'avg_review_length': stype.numerical,
        'pct_verified_reviews': stype.numerical,
        'product_bias': stype.numerical,
        'avg_reviewer_num_reviews': stype.numerical,
        'avg_reviewer_total_spent': stype.numerical,
        'avg_reviewer_avg_price': stype.numerical,
        'avg_reviewer_avg_rating': stype.numerical,
        'avg_reviewer_std_rating': stype.numerical
    },

    'rel-hm-sales': {
        'article_id': stype.numerical,
        'timestamp': stype.timestamp,
        'sales': stype.numerical,
        'week_of_year': stype.categorical,
        'month_of_year': stype.categorical,
        'day_of_month': stype.categorical,
        'department_no': stype.categorical,
        'section_no': stype.categorical,
        'perceived_colour_master_id': stype.categorical,
        'num_sales_1_weeks_ago': stype.numerical,
        'sold_amount_1_weeks_ago': stype.numerical,
        'avg_price_1_weeks_ago': stype.numerical,
        'num_customers_1_weeks_ago': stype.numerical,
        'avg_buyer_age_1_weeks_ago': stype.numerical,
        'avg_monthly_purchase_amount_1_weeks_ago': stype.numerical,
        'avg_monthly_purchase_count_1_weeks_ago': stype.numerical,
        'avg_weeks_since_last_purchase_1_weeks_ago': stype.numerical,
        'num_sales_2_weeks_ago': stype.numerical,
        'sold_amount_2_weeks_ago': stype.numerical,
        'avg_price_2_weeks_ago': stype.numerical,
        'num_customers_2_weeks_ago': stype.numerical,
        'avg_buyer_age_2_weeks_ago': stype.numerical,
        'avg_monthly_purchase_amount_2_weeks_ago': stype.numerical,
        'avg_monthly_purchase_count_2_weeks_ago': stype.numerical,
        'avg_weeks_since_last_purchase_2_weeks_ago': stype.numerical,
        'num_sales_3_weeks_ago': stype.numerical,
        'sold_amount_3_weeks_ago': stype.numerical,
        'avg_price_3_weeks_ago': stype.numerical,
        'num_customers_3_weeks_ago': stype.numerical,
        'avg_buyer_age_3_weeks_ago': stype.numerical,
        'avg_monthly_purchase_amount_3_weeks_ago': stype.numerical,
        'avg_monthly_purchase_count_3_weeks_ago': stype.numerical,
        'avg_weeks_since_last_purchase_3_weeks_ago': stype.numerical,
        'num_sales_4_weeks_ago': stype.numerical,
        'sold_amount_4_weeks_ago': stype.numerical,
        'avg_price_4_weeks_ago': stype.numerical,
        'num_customers_4_weeks_ago': stype.numerical,
        'avg_buyer_age_4_weeks_ago': stype.numerical,
        'avg_monthly_purchase_amount_4_weeks_ago': stype.numerical,
        'avg_monthly_purchase_count_4_weeks_ago': stype.numerical,
        'avg_weeks_since_last_purchase_4_weeks_ago': stype.numerical,
        'num_sales_5_weeks_ago': stype.numerical,
        'sold_amount_5_weeks_ago': stype.numerical,
        'avg_price_5_weeks_ago': stype.numerical,
        'num_customers_5_weeks_ago': stype.numerical,
        'avg_buyer_age_5_weeks_ago': stype.numerical,
        'avg_monthly_purchase_amount_5_weeks_ago': stype.numerical,
        'avg_monthly_purchase_count_5_weeks_ago': stype.numerical,
        'avg_weeks_since_last_purchase_5_weeks_ago': stype.numerical
    },

    'rel-hm-churn': {
        'customer_id': stype.numerical,
        'timestamp': stype.timestamp,
        'churn': stype.categorical,
        'week_of_year': stype.categorical,
        'month_of_year': stype.categorical,
        'day_of_month': stype.categorical,
        'age': stype.numerical,
        'fn_not_null': stype.categorical,
        'is_active': stype.categorical,
        'club_member_status': stype.categorical,
        'fashion_news_frequency': stype.categorical,
        'total_purchase_count': stype.numerical,
        'total_purchase_amount': stype.numerical,
        'avg_purchase_price': stype.numerical,
        'total_unique_articles_purchased': stype.numerical,
        'prop_sales_channel_2': stype.numerical,
        'modal_dept_no': stype.categorical,
        'modal_section_no': stype.categorical,
        'modal_color_id': stype.categorical,
        'num_purchases_1_weeks_ago': stype.numerical,
        'purchased_amount_1_weeks_ago': stype.numerical,
        'avg_purchase_price_1_weeks_ago': stype.numerical,
        'num_unique_articles_purchased_1_weeks_ago': stype.numerical,
        'prop_sales_channel_2_1_weeks_ago': stype.numerical,
        'avg_monthly_sales_amount_1_weeks_ago': stype.numerical,
        'avg_monthly_sales_count_1_weeks_ago': stype.numerical,
        'avg_days_since_last_sale_1_weeks_ago': stype.numerical,
        'modal_dept_no_1_weeks_ago': stype.categorical,
        'modal_section_no_1_weeks_ago': stype.categorical,
        'modal_color_id_1_weeks_ago': stype.categorical,
        'num_purchases_2_weeks_ago': stype.numerical,
        'purchased_amount_2_weeks_ago': stype.numerical,
        'avg_purchase_price_2_weeks_ago': stype.numerical,
        'num_unique_articles_purchased_2_weeks_ago': stype.numerical,
        'prop_sales_channel_2_2_weeks_ago': stype.numerical,
        'avg_monthly_sales_amount_2_weeks_ago': stype.numerical,
        'avg_monthly_sales_count_2_weeks_ago': stype.numerical,
        'avg_days_since_last_sale_2_weeks_ago': stype.numerical,
        'modal_dept_no_2_weeks_ago': stype.categorical,
        'modal_section_no_2_weeks_ago': stype.categorical,
        'modal_color_id_2_weeks_ago': stype.categorical,
        'num_purchases_3_weeks_ago': stype.numerical,
        'purchased_amount_3_weeks_ago': stype.numerical,
        'avg_purchase_price_3_weeks_ago': stype.numerical,
        'num_unique_articles_purchased_3_weeks_ago': stype.numerical,
        'prop_sales_channel_2_3_weeks_ago': stype.numerical,
        'avg_monthly_sales_amount_3_weeks_ago': stype.numerical,
        'avg_monthly_sales_count_3_weeks_ago': stype.numerical,
        'avg_days_since_last_sale_3_weeks_ago': stype.numerical,
        'modal_dept_no_3_weeks_ago': stype.categorical,
        'modal_section_no_3_weeks_ago': stype.categorical,
        'modal_color_id_3_weeks_ago': stype.categorical,
        'num_purchases_4_weeks_ago': stype.numerical,
        'purchased_amount_4_weeks_ago': stype.numerical,
        'avg_purchase_price_4_weeks_ago': stype.numerical,
        'num_unique_articles_purchased_4_weeks_ago': stype.numerical,
        'prop_sales_channel_2_4_weeks_ago': stype.numerical,
        'avg_monthly_sales_amount_4_weeks_ago': stype.numerical,
        'avg_monthly_sales_count_4_weeks_ago': stype.numerical,
        'avg_days_since_last_sale_4_weeks_ago': stype.numerical,
        'modal_dept_no_4_weeks_ago': stype.categorical,
        'modal_section_no_4_weeks_ago': stype.categorical,
        'modal_color_id_4_weeks_ago': stype.categorical,
        'num_purchases_5_weeks_ago': stype.numerical,
        'purchased_amount_5_weeks_ago': stype.numerical,
        'avg_purchase_price_5_weeks_ago': stype.numerical,
        'num_unique_articles_purchased_5_weeks_ago': stype.numerical,
        'prop_sales_channel_2_5_weeks_ago': stype.numerical,
        'avg_monthly_sales_amount_5_weeks_ago': stype.numerical,
        'avg_monthly_sales_count_5_weeks_ago': stype.numerical,
        'avg_days_since_last_sale_5_weeks_ago': stype.numerical,
        'modal_dept_no_5_weeks_ago': stype.categorical,
        'modal_section_no_5_weeks_ago': stype.categorical,
        'modal_color_id_5_weeks_ago': stype.categorical
    },

    'rel-f1-position': {
        'driverId': stype.numerical,
        'date': stype.timestamp,
        'position': stype.numerical,
        'week_of_year': stype.categorical,
        'driver_ref': stype.categorical,
        'driver_age': stype.numerical,
        'driver_nationality': stype.categorical,
        'driver_position': stype.numerical,
        'driver_points': stype.numerical,
        'driver_wins': stype.numerical,
        'driver_points_lag': stype.numerical,
        'driver_points_lead': stype.numerical,
        'days_since_last_race': stype.numerical,
        'constructor_ref': stype.categorical,
        'constructor_nationality': stype.categorical,
        'constructor_position': stype.numerical,
        'constructor_points': stype.numerical,
        'constructor_wins': stype.numerical,
        'constructor_points_lag': stype.numerical,
        'constructor_points_lead': stype.numerical,
        'position_diff': stype.numerical,
        'points_ratio': stype.numerical,
        'wins_ratio': stype.numerical,
        'past_1_driver_position': stype.numerical,
        'past_1_driver_points': stype.numerical,
        'past_1_driver_grid': stype.numerical,
        'past_1_position_gain': stype.numerical,
        'past_1_driver_rank': stype.numerical,
        'past_1_pct_laps_completed': stype.numerical,
        'past_1_dnf': stype.categorical,
        'past_1_constructor_points': stype.numerical,
        'upcoming_1_round': stype.categorical,
        'upcoming_1_circuit_id': stype.categorical,
        'past_2_driver_position': stype.numerical,
        'past_2_driver_points': stype.numerical,
        'past_2_driver_grid': stype.numerical,
        'past_2_position_gain': stype.numerical,
        'past_2_driver_rank': stype.numerical,
        'past_2_pct_laps_completed': stype.numerical,
        'past_2_dnf': stype.categorical,
        'past_2_constructor_points': stype.numerical,
        'upcoming_2_round': stype.categorical,
        'upcoming_2_circuit_id': stype.categorical,
        'past_3_driver_position': stype.numerical,
        'past_3_driver_points': stype.numerical,
        'past_3_driver_grid': stype.numerical,
        'past_3_position_gain': stype.numerical,
        'past_3_driver_rank': stype.numerical,
        'past_3_pct_laps_completed': stype.numerical,
        'past_3_dnf': stype.categorical,
        'past_3_constructor_points': stype.numerical,
        'upcoming_3_round': stype.categorical,
        'upcoming_3_circuit_id': stype.categorical
    },

    'rel-f1-dnf': {
        'driverId': stype.numerical,
        'date': stype.timestamp,
        'did_not_finish': stype.categorical,
        'week_of_year': stype.categorical,
        'driver_ref': stype.categorical,
        'driver_age': stype.numerical,
        'driver_nationality': stype.categorical,
        'driver_position': stype.numerical,
        'driver_points': stype.numerical,
        'driver_wins': stype.numerical,
        'driver_points_lag': stype.numerical,
        'driver_points_lead': stype.numerical,
        'days_since_last_race': stype.numerical,
        'constructor_ref': stype.categorical,
        'constructor_nationality': stype.categorical,
        'constructor_position': stype.numerical,
        'constructor_points': stype.numerical,
        'constructor_wins': stype.numerical,
        'constructor_points_lag': stype.numerical,
        'constructor_points_lead': stype.numerical,
        'position_diff': stype.numerical,
        'points_ratio': stype.numerical,
        'wins_ratio': stype.numerical,
        'past_1_driver_position': stype.numerical,
        'past_1_driver_points': stype.numerical,
        'past_1_driver_grid': stype.numerical,
        'past_1_position_gain': stype.numerical,
        'past_1_driver_rank': stype.numerical,
        'past_1_pct_laps_completed': stype.numerical,
        'past_1_dnf': stype.categorical,
        'past_1_constructor_points': stype.numerical,
        'upcoming_1_round': stype.categorical,
        'upcoming_1_circuit_id': stype.categorical,
        'past_2_driver_position': stype.numerical,
        'past_2_driver_points': stype.numerical,
        'past_2_driver_grid': stype.numerical,
        'past_2_position_gain': stype.numerical,
        'past_2_driver_rank': stype.numerical,
        'past_2_pct_laps_completed': stype.numerical,
        'past_2_dnf': stype.categorical,
        'past_2_constructor_points': stype.numerical,
        'upcoming_2_round': stype.categorical,
        'upcoming_2_circuit_id': stype.categorical,
        'past_3_driver_position': stype.numerical,
        'past_3_driver_points': stype.numerical,
        'past_3_driver_grid': stype.numerical,
        'past_3_position_gain': stype.numerical,
        'past_3_driver_rank': stype.numerical,
        'past_3_pct_laps_completed': stype.numerical,
        'past_3_dnf': stype.categorical,
        'past_3_constructor_points': stype.numerical,
        'upcoming_3_round': stype.categorical,
        'upcoming_3_circuit_id': stype.categorical
    },

    'rel-f1-qualifying': {
        'driverId': stype.numerical,
        'date': stype.timestamp,
        'qualifying': stype.categorical,
        'week_of_year': stype.categorical,
        'driver_ref': stype.categorical,
        'driver_age': stype.numerical,
        'driver_nationality': stype.categorical,
        'driver_position': stype.numerical,
        'driver_points': stype.numerical,
        'driver_wins': stype.numerical,
        'driver_points_lag': stype.numerical,
        'driver_points_lead': stype.numerical,
        'days_since_last_race': stype.numerical,
        'constructor_ref': stype.categorical,
        'constructor_nationality': stype.categorical,
        'constructor_position': stype.numerical,
        'constructor_points': stype.numerical,
        'constructor_wins': stype.numerical,
        'constructor_points_lag': stype.numerical,
        'constructor_points_lead': stype.numerical,
        'position_diff': stype.numerical,
        'points_ratio': stype.numerical,
        'wins_ratio': stype.numerical,
        'past_1_driver_position': stype.numerical,
        'past_1_driver_points': stype.numerical,
        'past_1_driver_grid': stype.numerical,
        'past_1_position_gain': stype.numerical,
        'past_1_driver_rank': stype.numerical,
        'past_1_pct_laps_completed': stype.numerical,
        'past_1_dnf': stype.categorical,
        'past_1_constructor_points': stype.numerical,
        'upcoming_1_round': stype.categorical,
        'upcoming_1_circuit_id': stype.categorical,
        'past_2_driver_position': stype.numerical,
        'past_2_driver_points': stype.numerical,
        'past_2_driver_grid': stype.numerical,
        'past_2_position_gain': stype.numerical,
        'past_2_driver_rank': stype.numerical,
        'past_2_pct_laps_completed': stype.numerical,
        'past_2_dnf': stype.categorical,
        'past_2_constructor_points': stype.numerical,
        'upcoming_2_round': stype.categorical,
        'upcoming_2_circuit_id': stype.categorical,
        'past_3_driver_position': stype.numerical,
        'past_3_driver_points': stype.numerical,
        'past_3_driver_grid': stype.numerical,
        'past_3_position_gain': stype.numerical,
        'past_3_driver_rank': stype.numerical,
        'past_3_pct_laps_completed': stype.numerical,
        'past_3_dnf': stype.categorical,
        'past_3_constructor_points': stype.numerical,
        'upcoming_3_round': stype.categorical,
        'upcoming_3_circuit_id': stype.categorical
    }
}
