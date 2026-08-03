-- Add research relevance and behavioral extraction fields to analyzed_reviews.
-- Run in Supabase SQL editor before enabling full production writes for these fields.

alter table public.analyzed_reviews
  add column if not exists research_relevance text,
  add column if not exists feedback_domain text,
  add column if not exists shopping_mission text,
  add column if not exists habit_signal text,
  add column if not exists current_category text,
  add column if not exists category_tried text,
  add column if not exists discovery_method text,
  add column if not exists exploration_barrier text,
  add column if not exists purchase_trigger text,
  add column if not exists information_needed text,
  add column if not exists trust_signal text,
  add column if not exists perceived_risk text,
  add column if not exists workaround text,
  add column if not exists user_context text,
  add column if not exists unmet_need_detail text,
  add column if not exists evidence_quote text,
  add column if not exists classification_confidence text;

alter table public.analyzed_reviews
  drop constraint if exists analyzed_reviews_research_relevance_check;
alter table public.analyzed_reviews
  add constraint analyzed_reviews_research_relevance_check
  check (research_relevance is null or research_relevance in (
    'discovery_relevant',
    'indirectly_relevant',
    'operational_only',
    'irrelevant'
  ));

alter table public.analyzed_reviews
  drop constraint if exists analyzed_reviews_feedback_domain_check;
alter table public.analyzed_reviews
  add constraint analyzed_reviews_feedback_domain_check
  check (feedback_domain is null or feedback_domain in (
    'category_discovery',
    'shopping_habit',
    'purchase_mission',
    'trust_and_quality',
    'product_information',
    'price_and_value',
    'recommendation',
    'assortment',
    'impulse_purchase',
    'workaround',
    'delivery',
    'refund',
    'customer_support',
    'app_technical',
    'other'
  ));

alter table public.analyzed_reviews
  drop constraint if exists analyzed_reviews_classification_confidence_check;
alter table public.analyzed_reviews
  add constraint analyzed_reviews_classification_confidence_check
  check (classification_confidence is null or classification_confidence in ('low', 'medium', 'high'));
