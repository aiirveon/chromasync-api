-- Run this in Supabase SQL Editor
-- Table: stories

create table if not exists stories (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  format text not null check (format in ('film', 'short_story')),
  raw_idea text,
  logline text,
  logline_label text,
  character_name text,
  character_lie text,
  character_want text,
  character_need text,
  save_the_cat_scene text,
  save_the_cat_framing text,
  beats jsonb default '[]'::jsonb,   -- array of { beat_number, beat_name, answer }
  stage integer not null default 0,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Row level security
alter table stories enable row level security;

create policy "Users can manage their own stories"
  on stories for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- Auto-update updated_at
create or replace function update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create trigger stories_updated_at
  before update on stories
  for each row execute function update_updated_at_column();

-- If table already exists, just add the beats column:
-- alter table stories add column if not exists beats jsonb default '[]'::jsonb;
