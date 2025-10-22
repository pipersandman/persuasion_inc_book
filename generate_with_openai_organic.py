"""Book generation with truly organic flow - all characteristics woven naturally."""
import openai
import json
import sqlite3
import time
import os
from typing import Dict, List
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Set API key
openai.api_key = os.getenv('OPENAI_API_KEY')

class PersuasionIncGenerator:
    """Generate Persuasion Inc with organic, natural flow."""
    
    def __init__(self):
        # Verify API key is loaded
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
        
        print(f"API key loaded: {openai.api_key[:20]}...")
        
        # Load style profile
        with open('out/writing_style_profile.json', 'r', encoding='utf-8') as f:
            self.style = json.load(f)
        
        # Load book outline
        with open('book_outline.json', 'r', encoding='utf-8') as f:
            self.book = json.load(f)
        
        # Database connection
        self.conn = sqlite3.connect('out/conversations.db')
        
        # Track generated content
        self.previous_chapters = []
    
    def mine_conversations(self, topic: str, subtopics: List[str], limit: int = 10) -> List[Dict]:
        """Find relevant conversations for this chapter."""
        relevant = []
        
        # Search for main topic and subtopics
        search_terms = [topic] + subtopics
        search_terms = [term.lower().split()[:2] for term in search_terms]
        search_terms = [word for sublist in search_terms for word in sublist]
        
        for term in set(search_terms):
            query = """
                SELECT title, full_text, created_at
                FROM conversations
                WHERE LOWER(full_text) LIKE LOWER(?)
                ORDER BY created_at DESC
                LIMIT ?
            """
            results = self.conn.execute(query, (f'%{term}%', limit)).fetchall()
            
            for title, text, date in results:
                term_index = text.lower().find(term)
                if term_index != -1:
                    start = max(0, term_index - 200)
                    end = min(len(text), term_index + 500)
                    excerpt = text[start:end]
                    
                    relevant.append({
                        'title': title,
                        'excerpt': excerpt,
                        'date': date,
                        'term': term
                    })
        
        # Deduplicate
        seen_titles = set()
        unique = []
        for conv in relevant:
            if conv['title'] not in seen_titles:
                seen_titles.add(conv['title'])
                unique.append(conv)
        
        return unique[:limit]
    
    def build_organic_prompt(self, chapter: Dict, relevant_convs: List[Dict]) -> str:
        """Create a prompt that encourages natural flow and organic style integration."""
        
        # Extract insights from conversations
        conv_insights = ""
        if relevant_convs:
            conv_insights = "\n\nRelevant insights from past conversations to weave in naturally:\n"
            for conv in relevant_convs[:5]:
                conv_insights += f"- From '{conv['title']}': {conv['excerpt'][:200]}...\n"
        
        # Build context from previous chapters
        previous_context = ""
        if self.previous_chapters:
            previous_context = "\n\nPrevious chapters have explored (build on these without repeating):\n"
            for i, prev in enumerate(self.previous_chapters[-2:]):
                # Get first substantial paragraph
                paragraphs = [p for p in prev.split('\n\n') if len(p) > 100]
                if paragraphs:
                    previous_context += f"- Chapter {i}: {paragraphs[0][:250]}...\n"
        
        return f"""Write a complete chapter for 'Persuasion, Inc.' titled: {chapter['title']}

CORE THEMES TO EXPLORE:
{chr(10).join(f'- {topic}' for topic in chapter['subtopics'])}

KEY ARGUMENTS TO DEVELOP (weave throughout naturally):
{chr(10).join(f'- {point}' for point in chapter['key_points'])}

{conv_insights}

{previous_context}

VOICE & STYLE:
You're channeling a blend of: Orwell's moral clarity + Rushkoff's rebellious academic + Zuboff's depth + David Foster Wallace's self-aware intelligence + Jon Ronson's wry wit + Chuck Klosterman's philosophical playfulness.

Write as someone who sees both the horror and dark comedy of our technological present. Authoritative but self-aware. The smart friend who's been reading too much Orwell and watching too much Black Mirror.

STRUCTURE (but let it flow organically - these aren't section headers):

START WITH A HOOK that immediately grabs attention:
- Could be: a weird anecdote, a dystopian scenario pushed to absurdity, an uncomfortable observation, a satirical thought experiment
- Make them stop and think "wait, what?"
- Example approaches: "Imagine if..." / "Consider that..." / "Here's something nobody talks about..."

THEN FLOW NATURALLY between these modes (don't label them, just let them emerge):

SERIOUS ANALYSIS: Ground your arguments in reality
- Reference behavioral economics, media theory, surveillance capitalism concepts
- Include data points, historical patterns, psychological mechanisms
- Channel Zuboff's depth and Rushkoff's systemic critique
- Make complex ideas accessible to smart general readers

DARK HUMOR & SATIRE: Let absurdity reveal truth
- When systems are ridiculous, acknowledge it with dry wit
- Use hypothetical scenarios pushed to revealing extremes
- Channel Jon Ronson's wry observations and Charlie Brooker's darkness
- Black Mirror meets Chuck Klosterman's "what if?" thinking

ACADEMIC GROUNDING: Reference actual concepts without being dry
- Mention specific theories, thinkers, or studies naturally
- Translate complex ideas clearly
- Show you've done the homework but keep it conversational

REFLECTIVE INSIGHT: Build to awareness without preaching
- Help readers notice what they've been missing
- Connect dots between seemingly separate phenomena
- Leave them thinking differently, not despairing

END WITH SOMETHING THAT LINGERS:
- Not a neat bow, but a thought that haunts
- Set up curiosity for: {chapter.get('connections', ['the next chapter'])[0]}
- Orwell's urgency + Doctorow's warning, but nuanced

CRITICAL RULES:
1. NEVER use em-dashes (— or –). Use commas, colons, semicolons, or split into sentences.
2. Don't create section headers within the chapter. Let the prose flow.
3. Average sentence length around {self.style['statistics']['avg_sentence_length']:.0f} words, but vary naturally
4. Aesthetic: Minimalist dystopian corporate (think Severance, The Circle)
5. Perspective: First-person "we" (inclusive observer) or third-person analytical

Let the different tones serve the content organically:
- Serious when revealing mechanisms of control
- Satirical when exposing absurdities
- Academic when grounding claims
- Reflective when driving points home

The reader should feel like they're being let in on something important by someone who understands both the gravity and the absurdity of our situation.

TARGET: 2500-3500 words. Write the complete chapter as flowing prose with natural paragraph breaks, not rigid sections.
"""
    
    def generate_chapter(self, chapter_index: int, model: str = "gpt-4") -> str:
        """Generate a complete chapter with organic flow."""
        chapter = self.book['chapters'][chapter_index]
        print(f"\n📖 Generating: {chapter['title']}")
        print("=" * 60)
        
        # Mine conversations for relevant content
        print("🔍 Mining conversations for relevant content...")
        relevant = self.mine_conversations(
            chapter['title'],
            chapter['subtopics']
        )
        print(f"   Found {len(relevant)} relevant conversations")
        
        # Build the organic prompt
        prompt = self.build_organic_prompt(chapter, relevant)
        
        print(f"✍️  Generating chapter with organic flow...")
        print(f"    (This may take 1-2 minutes for GPT-4 to generate ~3000 words)")
        
        try:
            # Call OpenAI API with generous token limit for full chapter
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert writer creating 'Persuasion, Inc.' - a book about 
surveillance capitalism, algorithmic persuasion, and technological control. Your writing style blends:
- Orwell's moral urgency and clarity
- Zuboff's academic depth made accessible  
- Rushkoff's rebellious cultural criticism
- David Foster Wallace's intelligence and self-awareness
- Jon Ronson's wry investigative wit
- Chuck Klosterman's philosophical hypotheticals

You write serious analysis punctuated by dark humor. You ground arguments in behavioral economics 
and media theory while keeping it readable. You help people see systems of control they've missed 
while acknowledging the absurdity of it all. You NEVER use em-dashes (— or –)."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4500,  # Generous limit for 3000+ words
                presence_penalty=0.3,
                frequency_penalty=0.3
            )
            
            chapter_text = response['choices'][0]['message']['content']
            
            # Clean up
            chapter_text = self.clean_text(chapter_text)
            
            # Add title if not present
            if not chapter_text.startswith('#'):
                chapter_text = f"# {chapter['title']}\n\n{chapter_text}"
            
            # Track for context
            self.previous_chapters.append(chapter_text)
            
            # Save
            filename = f"generated_ch{chapter_index+1:02d}_{chapter['title'][:30].replace(' ', '_').replace(':', '')}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(chapter_text)
            
            print(f"💾 Saved: {filename}")
            print(f"📊 Chapter word count: {len(chapter_text.split())} words")
            
            return chapter_text
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️ Error generating chapter: {error_msg}")
            
            # If it's a 502 or network error, provide helpful message
            if "502" in error_msg or "Bad Gateway" in error_msg:
                print("\n   💡 This is a temporary OpenAI/Cloudflare server issue.")
                print("   Just run the command again and it should work.")
                return ""
            
            # For other errors, try fallback
            return self.generate_chapter_fallback(chapter, relevant, model)
    
    def generate_chapter_fallback(self, chapter: Dict, relevant_convs: List[Dict], model: str) -> str:
        """Fallback: generate in two parts if full generation fails."""
        print("   🔄 Attempting generation in two parts...")
        
        full_chapter = f"# {chapter['title']}\n\n"
        
        # Part 1: First half
        prompt1 = f"""Write the first 1500-2000 words for chapter '{chapter['title']}'.

Start with a compelling hook, then develop these themes: {', '.join(chapter['subtopics'][:2])}

Use the voice described: Orwell + Rushkoff + David Foster Wallace. Serious analysis with dark humor.
NO em-dashes. Flow naturally between analysis, satire, and insight."""
        
        try:
            response1 = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You're writing 'Persuasion, Inc.' with Orwell's clarity and dark humor."},
                    {"role": "user", "content": prompt1}
                ],
                temperature=0.8,
                max_tokens=2500,
                presence_penalty=0.3,
                frequency_penalty=0.3
            )
            
            part1 = self.clean_text(response1['choices'][0]['message']['content'])
            full_chapter += part1 + "\n\n"
            print("   ✓ Generated first part")
            time.sleep(3)
            
            # Part 2: Second half
            prompt2 = f"""Continue and conclude the chapter '{chapter['title']}'.

Build on what came before. Develop these remaining themes: {', '.join(chapter['subtopics'][2:])}

Ground arguments in behavioral economics or media theory. End with something that lingers.
NO em-dashes. Maintain the voice: analytical but wry, serious but self-aware."""
            
            response2 = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Continue writing 'Persuasion, Inc.' maintaining voice and style."},
                    {"role": "user", "content": prompt2}
                ],
                temperature=0.8,
                max_tokens=2500,
                presence_penalty=0.3,
                frequency_penalty=0.3
            )
            
            part2 = self.clean_text(response2['choices'][0]['message']['content'])
            full_chapter += part2
            print("   ✓ Generated second part")
            
            return full_chapter
            
        except Exception as e:
            print(f"   ✗ Fallback also failed: {str(e)}")
            return f"# {chapter['title']}\n\n[Chapter generation failed: {str(e)}]"
    
    def clean_text(self, text: str) -> str:
        """Remove em-dashes and clean formatting."""
        # Remove em-dashes
        text = text.replace("—", ", ")
        text = text.replace("–", ", ")
        text = re.sub(r'\s+—\s+', ', ', text)
        text = re.sub(r'\s+–\s+', ', ', text)
        
        # Remove any rigid section headers that might have been generated
        text = re.sub(r'^##\s*(Hook|Current Realities|Satirical Turn|Academic Insight|Call to Awareness)\s*$', '', text, flags=re.MULTILINE)
        
        # Clean spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def generate_full_book(self, start_chapter: int = 0, model: str = "gpt-4"):
        """Generate the entire book."""
        print(f"\n🚀 STARTING BOOK GENERATION")
        print(f"Model: {model}")
        print(f"Chapters to generate: {len(self.book['chapters']) - start_chapter}")
        
        full_book = f"# {self.book['title']}\n\n"
        
        for i in range(start_chapter, len(self.book['chapters'])):
            chapter_text = self.generate_chapter(i, model)
            
            if chapter_text:  # Only add if generation succeeded
                full_book += chapter_text + "\n\n" + "="*80 + "\n\n"
                
                # Save complete book so far
                with open('persuasion_inc_full.md', 'w', encoding='utf-8') as f:
                    f.write(full_book)
                
                print(f"\n📚 Progress: {i+1}/{len(self.book['chapters'])} chapters complete")
            
            # Pause between chapters
            if i < len(self.book['chapters']) - 1:
                print("⏸️  Pausing 10 seconds before next chapter...")
                time.sleep(10)
        
        print(f"\n🎉 BOOK GENERATION COMPLETE!")
        print(f"📁 Full book saved as: persuasion_inc_full.md")
        print(f"📊 Total words: {len(full_book.split())}")
        
        self.conn.close()
        return full_book


if __name__ == "__main__":
    import sys
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
        sys.exit(1)
    
    try:
        generator = PersuasionIncGenerator()
        
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            # Test mode: just generate introduction
            print("🧪 TEST MODE: Generating Introduction only")
            intro = generator.generate_chapter(0, model="gpt-4")
            
            if intro:
                print("\n✅ Test complete! Check the generated file.")
                print("\nThe chapter should now flow naturally without rigid section headers.")
                print("All writing characteristics (serious/satirical/academic/reflective) should")
                print("emerge organically as the content demands, not in formula sections.")
            else:
                print("\n⚠️  Generation failed. If you got a 502 error, just run it again.")
        else:
            # Full generation
            print("📚 FULL BOOK GENERATION MODE")
            print("This will generate all chapters using GPT-4 with organic flow")
            print("Estimated cost: $30-40")
            print("Estimated time: 40-60 minutes")
            
            confirm = input("\nProceed? (yes/no): ")
            if confirm.lower() == 'yes':
                generator.generate_full_book(start_chapter=0, model="gpt-4")
            else:
                print("Cancelled. Run with 'test' argument to generate just the introduction.")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has OPENAI_API_KEY=sk-...")
        print("2. Make sure you've run: pip install openai==0.28.1")
        print("3. Verify your API key is valid at platform.openai.com")
