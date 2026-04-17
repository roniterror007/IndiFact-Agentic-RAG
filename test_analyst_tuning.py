#!/usr/bin/env python
"""
Quick test to validate analyst prompt tuning without running full harness.
Tests 3 queries with known answers to see if VERIFIED rate improved.
"""
import json
import os
import sys
from pathlib import Path

os.environ['LLM_BACKEND'] = 'ollama'
os.environ['LLM_MODEL'] = 'llama3.2:1b'

from src.retrieval.indexer import IndicHybridIndexer
from src.data.curation import load_jsonl_documents
from src.agents.personas import build_default_personas

def test_analyst_tuning():
    """Test if analyst produces VERIFIED for straightforward queries."""
    
    # Load corpus
    corpus_path = Path('datasets/sample_corpus.jsonl')
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found")
        return False
    
    corpus_docs = load_jsonl_documents(str(corpus_path))
    print(f"Loaded {len(corpus_docs)} corpus documents\n")
    
    # Create indexer
    indexer = IndicHybridIndexer(collection_name='analyst_tuning_validation')
    indexer.add_documents(corpus_docs)
    
    # Build personas
    personas = build_default_personas(indexer.hybrid_search)
    analyst = personas['analyst']
    searcher = personas['searcher']
    
    # Test queries
    test_cases = [
        {
            'query': 'Did GST launch in India on 1 July 2017?',
            'expected': 'VERIFIED',
            'reason': 'GST document explicitly states this date'
        },
        {
            'query': 'When did Chandrayaan-3 land on the moon?',
            'expected': 'VERIFIED',
            'reason': 'Chandrayaan-3 document provides landing date'
        },
        {
            'query': 'Did India win the 2020 FIFA World Cup?',
            'expected': 'KNOWLEDGE_GAP',
            'reason': 'World Cup document explicitly says India did NOT win'
        }
    ]
    
    results = []
    verified_count = 0
    knowledge_gap_count = 0
    
    print("=" * 70)
    print("ANALYST PROMPT TUNING VALIDATION TEST")
    print("=" * 70)
    
    for i, test in enumerate(test_cases, 1):
        query = test['query']
        expected_status = test['expected']
        reason = test['reason']
        
        # Retrieve evidence
        search_result = searcher.run(query=query, k=5)
        documents = search_result.get('documents', [])
        
        # Analyze
        analyst_output = analyst.run(query=query, documents=documents)
        actual_status = analyst_output.get('status', 'UNKNOWN')
        answer = analyst_output.get('answer', '')[:60]
        
        # Track
        if actual_status == 'VERIFIED':
            verified_count += 1
        elif actual_status == 'KNOWLEDGE_GAP':
            knowledge_gap_count += 1
        
        # Display
        print(f"\nTest {i}:")
        print(f"  Query: {query}")
        print(f"  Expected: {expected_status}")
        print(f"  Actual: {actual_status}")
        print(f"  Match: {'✓ PASS' if actual_status == expected_status else '✗ FAIL'}")
        print(f"  Reason: {reason}")
        if answer:
            print(f"  Answer: {answer}...")
        
        results.append({
            'query': query,
            'expected': expected_status,
            'actual': actual_status,
            'passed': actual_status == expected_status
        })
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r['passed'])
    print(f"Tests Passed: {passed}/{len(results)}")
    print(f"VERIFIED outputs: {verified_count}")
    print(f"KNOWLEDGE_GAP outputs: {knowledge_gap_count}")
    
    success = passed == len(results)
    if success:
        print("\n✓ SUCCESS: Analyst tuning is working correctly!")
    else:
        print("\n⚠ PARTIAL: Some tests failed. Analyst may need further tuning.")
    
    return success

if __name__ == '__main__':
    try:
        success = test_analyst_tuning()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
