from torch_frame import stype

task_to_stypes = {
    'rel-stackex-engage': {
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

    'rel-stackex-badges': {
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

    'rel-stackex-votes': {
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
    }
}
