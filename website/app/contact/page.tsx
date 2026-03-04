'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Mail,
  Github,
  MessageSquare,
  Send,
  Check,
  AlertCircle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

interface FormData {
  name: string;
  email: string;
  category: string;
  subject: string;
  message: string;
}

const categories = [
  { value: 'support', label: 'Technical Support' },
  { value: 'partnership', label: 'Partnership Inquiry' },
  { value: 'research', label: 'Research Collaboration' },
  { value: 'feedback', label: 'Product Feedback' },
  { value: 'other', label: 'General Question' },
];

const faqs = [
  {
    question: 'Is ProteinProcessIO free to use?',
    answer:
      'Yes, ProteinProcessIO is free for research and academic use. Commercial licensing options are available for industrial applications.',
  },
  {
    question: 'What are the system requirements?',
    answer:
      'ProteinProcessIO runs on Windows 10+, macOS 11+, and Linux. For GPU acceleration, an NVIDIA GPU with CUDA 11.0+ is recommended. See the Download page for full requirements.',
  },
  {
    question: 'Can I use ProteinProcessIO for commercial projects?',
    answer:
      'Please contact us for commercial licensing options. We offer flexible terms for industrial and commercial use cases.',
  },
  {
    question: 'How accurate are the simulations?',
    answer:
      'Our physics models are validated against experimental data from NRC Canada pilot-scale equipment. The pretreatment module has been calibrated against real GP-15 machine data.',
  },
  {
    question: 'Is training available?',
    answer:
      'Yes, we offer documentation, tutorials, and webinar sessions. For custom training programs, please contact us to discuss your needs.',
  },
  {
    question: 'How can I contribute to the project?',
    answer:
      'We welcome contributions! Check out our GitHub repository for contribution guidelines, or contact us to discuss research collaboration opportunities.',
  },
];

export default function ContactPage() {
  const [formData, setFormData] = useState<FormData>({
    name: '',
    email: '',
    category: 'support',
    subject: '',
    message: '',
  });
  const [status, setStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('submitting');

    // Simulate form submission
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // In production, you would send this to your backend
    console.log('Form submitted:', formData);
    setStatus('success');

    // Reset form after success
    setTimeout(() => {
      setFormData({
        name: '',
        email: '',
        category: 'support',
        subject: '',
        message: '',
      });
      setStatus('idle');
    }, 3000);
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  return (
    <main className="pt-24">
      {/* Hero Section */}
      <section className="section pb-12">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-3xl mx-auto text-center"
          >
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6">
              Get in <span className="gradient-text">Touch</span>
            </h1>
            <p className="text-xl text-text-secondary">
              Have questions, feedback, or interested in collaboration? We'd
              love to hear from you.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Contact Form & Info */}
      <section className="section pt-0">
        <div className="container-custom">
          <div className="grid lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {/* Contact Form */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="lg:col-span-2"
            >
              <div className="glass-card p-8">
                <h2 className="text-2xl font-bold text-white mb-6">
                  Send a Message
                </h2>

                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Name & Email Row */}
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <label
                        htmlFor="name"
                        className="block text-sm font-medium text-text-secondary mb-2"
                      >
                        Name
                      </label>
                      <input
                        type="text"
                        id="name"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-3 rounded-lg bg-bg-surface border border-white/10 text-white placeholder-text-muted focus:outline-none focus:border-accent transition-colors"
                        placeholder="Your name"
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="email"
                        className="block text-sm font-medium text-text-secondary mb-2"
                      >
                        Email
                      </label>
                      <input
                        type="email"
                        id="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-3 rounded-lg bg-bg-surface border border-white/10 text-white placeholder-text-muted focus:outline-none focus:border-accent transition-colors"
                        placeholder="you@example.com"
                      />
                    </div>
                  </div>

                  {/* Category */}
                  <div>
                    <label
                      htmlFor="category"
                      className="block text-sm font-medium text-text-secondary mb-2"
                    >
                      Category
                    </label>
                    <select
                      id="category"
                      name="category"
                      value={formData.category}
                      onChange={handleChange}
                      className="w-full px-4 py-3 rounded-lg bg-bg-surface border border-white/10 text-white focus:outline-none focus:border-accent transition-colors"
                    >
                      {categories.map((cat) => (
                        <option key={cat.value} value={cat.value}>
                          {cat.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Subject */}
                  <div>
                    <label
                      htmlFor="subject"
                      className="block text-sm font-medium text-text-secondary mb-2"
                    >
                      Subject
                    </label>
                    <input
                      type="text"
                      id="subject"
                      name="subject"
                      value={formData.subject}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 rounded-lg bg-bg-surface border border-white/10 text-white placeholder-text-muted focus:outline-none focus:border-accent transition-colors"
                      placeholder="How can we help?"
                    />
                  </div>

                  {/* Message */}
                  <div>
                    <label
                      htmlFor="message"
                      className="block text-sm font-medium text-text-secondary mb-2"
                    >
                      Message
                    </label>
                    <textarea
                      id="message"
                      name="message"
                      value={formData.message}
                      onChange={handleChange}
                      required
                      rows={5}
                      className="w-full px-4 py-3 rounded-lg bg-bg-surface border border-white/10 text-white placeholder-text-muted focus:outline-none focus:border-accent transition-colors resize-none"
                      placeholder="Tell us more about your inquiry..."
                    />
                  </div>

                  {/* Submit Button */}
                  <button
                    type="submit"
                    disabled={status === 'submitting' || status === 'success'}
                    className="btn-accent w-full py-4 text-base disabled:opacity-50"
                  >
                    {status === 'idle' && (
                      <>
                        <Send className="w-5 h-5" />
                        Send Message
                      </>
                    )}
                    {status === 'submitting' && (
                      <>
                        <div className="w-5 h-5 border-2 border-bg-dark/30 border-t-bg-dark rounded-full animate-spin" />
                        Sending...
                      </>
                    )}
                    {status === 'success' && (
                      <>
                        <Check className="w-5 h-5" />
                        Message Sent!
                      </>
                    )}
                    {status === 'error' && (
                      <>
                        <AlertCircle className="w-5 h-5" />
                        Error - Try Again
                      </>
                    )}
                  </button>
                </form>
              </div>
            </motion.div>

            {/* Contact Info Sidebar */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-6"
            >
              {/* Email Card */}
              <div className="glass-card p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                    <Mail className="w-6 h-6 text-accent" />
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-text-muted">Email</h3>
                    <a
                      href="mailto:contact@proteinprocessio.com"
                      className="text-white hover:text-accent transition-colors"
                    >
                      contact@proteinprocessio.com
                    </a>
                  </div>
                </div>
              </div>

              {/* GitHub Card */}
              <div className="glass-card p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                    <Github className="w-6 h-6 text-accent" />
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-text-muted">
                      GitHub Issues
                    </h3>
                    <a
                      href="https://github.com/mvgill/proteinprocessio/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-white hover:text-accent transition-colors"
                    >
                      Report a Bug
                    </a>
                  </div>
                </div>
              </div>

              {/* Response Time */}
              <div className="glass-card p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                    <MessageSquare className="w-6 h-6 text-accent" />
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-text-muted">
                      Response Time
                    </h3>
                    <p className="text-white">Within 2 business days</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="section bg-bg-surface/50">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="max-w-3xl mx-auto"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white text-center mb-12">
              Frequently Asked Questions
            </h2>

            <div className="space-y-4">
              {faqs.map((faq, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.05 }}
                  className="glass-card overflow-hidden"
                >
                  <button
                    onClick={() =>
                      setExpandedFaq(expandedFaq === index ? null : index)
                    }
                    className="w-full flex items-center justify-between p-6 text-left"
                  >
                    <span className="font-medium text-white pr-4">
                      {faq.question}
                    </span>
                    {expandedFaq === index ? (
                      <ChevronUp className="w-5 h-5 text-accent flex-shrink-0" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-text-muted flex-shrink-0" />
                    )}
                  </button>
                  {expandedFaq === index && (
                    <div className="px-6 pb-6">
                      <p className="text-text-secondary">{faq.answer}</p>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
    </main>
  );
}
